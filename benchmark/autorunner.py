import argparse
import os
import yaml
import subprocess
import time

from orchestrator.benchmark import BenchmarkRunner
from orchestrator.client import HTTPModelWrapper
from src.experiment_results import create_run_dir, write_dataset_results
from src.metrics import HTRMetricsEvaluator



def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Blad podczas wczytywania pliku YAML: {exc}")
            return None

def run_docker_command(command_list):
    """Pomocnicza funkcja do wykonywania komend docker compose."""
    try:
        result = subprocess.run(
            ["docker", "compose"] + command_list,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Blad Dockera: {e.stderr}")
        return None

def start_service(service_name):
    run_docker_command(["up", "-d", service_name])
    time.sleep(5) 

def stop_service(service_name):
    print(f"--- Zatrzymuje serwis: {service_name} ---")
    run_docker_command(["stop", service_name])

def cleanup_all():
    print("--- Sprzatanie wszystkich serwisow ---")
    run_docker_command(["down"])


def init_wandb_run(experiment: dict, config: dict, run_name: str):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Brak pakietu wandb. Zainstaluj: pip install wandb") from exc

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("Brak zmiennej WANDB_API_KEY w srodowisku.")

    wandb_project = experiment.get("project_name")
    if not wandb_project:
        raise RuntimeError("Brak experiment.wandb_project w experiments.yaml.")

    wandb_entity = experiment.get("wandb_entity")
    tags = experiment.get("tags") or []

    return wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        tags=tags,
        config=config,
    )


def build_timing_metrics(prediction_seconds: float, sample_count: int) -> dict:
    total_seconds = float(prediction_seconds)
    per_sample = total_seconds / sample_count if sample_count > 0 else 0.0
    throughput = sample_count / total_seconds if total_seconds > 0 else 0.0
    return {
        "prediction_seconds_total": total_seconds,
        "prediction_seconds_per_sample": per_sample,
        "prediction_throughput_samples_per_sec": throughput,
        "prediction_samples": int(sample_count),
    }


def build_wandb_payload(
    *,
    model_id: str,
    dataset_id: str,
    metrics_level: str,
    report: dict,
    timing: dict,
) -> dict:
    metrics = report.get("metrics", {}).get(metrics_level, {})
    prefix = f"{model_id}/{dataset_id}"

    payload = {
        f"{prefix}/metrics_level": metrics_level,
    }

    for key, value in metrics.items():
        payload[f"{prefix}/metrics/{key}"] = value

    for key, value in timing.items():
        payload[f"{prefix}/timing/{key}"] = value

    return payload

class AutoRunner:
    def __init__(self, config_path = "experiments.yaml"):
        self.experiment_config = load_config(config_path)
        if self.experiment_config is None:
            raise ValueError("Nie mozna wczytac konfiguracji eksperymentu.")
        
    def run(self, *, build_only: bool = False) -> None:
        experiment = self.experiment_config.get("experiment", {})
        settings = self.experiment_config.get("settings", {})
        print(f"Uruchamianie eksperymentu: {experiment.get('name', 'Unnamed')}")

        if build_only:
            print("--- Budowanie obrazow Docker ---")
            models = self.experiment_config.get("models", [])
            for model in models:
                if model.get("enabled", True):
                    print(f"Budowanie obrazu dla serwisu: {model['service_name']}")
                    run_docker_command(["build", model["service_name"]])
            print("Budowanie zakonczone.")
            return

        output_dir = settings.get("output_dir", "wyniki")
        experiment_name = experiment.get("name", "Unnamed")
        run_dir, run_name = create_run_dir(output_dir, experiment_name, self.experiment_config)

        wandb_run = None
        if settings.get("wandb_log", False):
            wandb_run = init_wandb_run(experiment, self.experiment_config, run_name)
        
        stop_after_run = settings.get("stop_service_after_run", True)
        metrics_evaluator = HTRMetricsEvaluator()

        try:
            if settings.get("auto_start_services", False) and settings.get("sequential", True):
                for model in self.experiment_config.get("models", []):
                    
                    if not model.get("enabled", True):
                        continue

                    print(f"\n=================================================")
                    print(f"Ewaluacja modelu: {model['id']}")
                    print(f"=================================================")

                    start_service(model["service_name"])
                    options = model.get("options", {})
                    
                    try:
                        try:
                            http_model = HTTPModelWrapper(
                                model_name=model["id"], 
                                base_url=model["base_url"], 
                                timeout_seconds=model["timeout"], 
                                options=options
                            )
                            
                            health = http_model.client.health()
                            print(f"Health check dla {model['service_name']}: {health}")
                            if health.get("status") != "ok":
                                raise RuntimeError(f"Serwis {model['service_name']} nie jest zdrowy: {health}")

                            load_info = http_model.load()
                            print(f"Load dla {model['service_name']}: {load_info}")
                            
                            runner = BenchmarkRunner(model=http_model)

                            for dataset in self.experiment_config.get("datasets", []):
                                print(f"--- Uruchamiam benchmark {model['id']} na zbiorze {dataset['id']} ---")
                                
                                try:
                                    results_df, prediction_seconds, sample_count = runner.run_with_timing(
                                        labels_csv=dataset["labels_csv"],
                                        images_dir=dataset["images_dir"],
                                        limit=dataset.get("limit", 1),
                                    )

                                    metrics_level = "word" if dataset.get("single_words", False) else "line"
                                    report = metrics_evaluator.build_report(
                                        results_df,
                                        model_name=model["id"],
                                        levels=(metrics_level,),
                                    )

                                    timing = build_timing_metrics(prediction_seconds, sample_count)
                                    report["dataset_id"] = dataset["id"]
                                    report["timing"] = timing

                                    write_dataset_results(
                                        run_dir=run_dir,
                                        model_id=model["id"],
                                        dataset_id=dataset["id"],
                                        results_df=results_df,
                                        summary=report,
                                    )

                                    if wandb_run is not None:
                                        log_payload = build_wandb_payload(
                                            model_id=model["id"],
                                            dataset_id=dataset["id"],
                                            metrics_level=metrics_level,
                                            report=report,
                                            timing=timing,
                                        )
                                        wandb_run.log(log_payload)
                                
                                except Exception as e_dataset:
                                    print(f"[BLAD DATASETU] Model: {model['id']}, dataset: {dataset['id']}. Szczegoly: {e_dataset}")
                        
                        except Exception as e_model:
                            print(f"[BLAD MODELU] Awaria {model['id']}. Pomijam ten model. Szczegoly: {e_model}")
                    
                    finally:
                        if stop_after_run:
                            stop_service(model["service_name"])
                    
                    print(f"cooldown {settings.get('cooldown_seconds', 5)}s")
                    time.sleep(settings.get("cooldown_seconds", 5))
        
        finally:
            if wandb_run is not None:
                wandb_run.finish()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autorunner benchmarku OCR")
    parser.add_argument("--build", action="store_true", help="Zbuduj obrazy Docker i zakoncz")
    args = parser.parse_args()

    autorunner = AutoRunner()
    autorunner.run(build_only=args.build)