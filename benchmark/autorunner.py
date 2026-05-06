from pathlib import Path
import sys
import yaml
import subprocess
import time

from orchestrator.benchmark import BenchmarkRunner
from orchestrator.client import HTTPModelWrapper



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
    print(f"--- Uruchamiam serwis: {service_name} ---")
    run_docker_command(["up", "-d", service_name])
    time.sleep(5) 

def stop_service(service_name):
    print(f"--- Zatrzymuje serwis: {service_name} ---")
    run_docker_command(["stop", service_name])

def cleanup_all():
    print("--- Sprzatanie wszystkich serwisow ---")
    run_docker_command(["down"])

class AutoRunner:
    def __init__(self, config_path = "experiments.yaml"):
        self.experiment_config = load_config(config_path)
        if self.experiment_config is None:
            raise ValueError("Nie mozna wczytac konfiguracji eksperymentu.")
        
    def run(self):
        experiment = self.experiment_config.get("experiment", {})
        settings = self.experiment_config.get("settings", {})
        print(f"Uruchamianie eksperymentu: {experiment.get('name', 'Unnamed')}")

        if settings.get("build_run", False):
            print("--- Budowanie obrazow Docker ---")
            run_docker_command(["build", "--no-cache"])
        
        if settings.get("auto_start_services", False) and settings.get("sequential", True):
            for model in self.experiment_config.get("models", []):
                if model.get("enabled", True):
                    start_service(model["service_name"])
                    options = model.get("options", {})
                    try:
                        # health check
                        http_model = HTTPModelWrapper(model_name=model["id"], base_url=model["base_url"], timeout_seconds=model["timeout"], options=options)
                        health = http_model.client.health()
                        print(f"Health check dla {model['service_name']}: {health}")
                        if health.get("status") != "ok":
                            raise RuntimeError(f"Serwis {model['service_name']} nie jest zdrowy: {health}")
                        
                        runner = BenchmarkRunner(model=http_model)

                        for dataset in self.experiment_config.get("datasets", []):
                            print(f"--- Uruchamiam benchmark dla modelu {model['id']} na zbiorze {dataset['id']} ---")
                            runner.run(
                                labels_csv=dataset["labels_csv"],
                                images_dir=dataset["images_dir"],
                                output_dir=settings.get("output_dir", "wyniki"),
                                limit=dataset.get("limit", 1),
                            )
                    finally:
                        stop_service(model["service_name"])
                    time.sleep(settings.get("cooldown_seconds", 5))
        


if __name__ == "__main__":
    autorunner = AutoRunner()
    autorunner.run()