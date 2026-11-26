#!/usr/bin/env python
"""Cały przebieg od zera na wybranym zbiorze testowym: od surowych predykcji benchmarku do paczki i tabel.

    python rebuild.py configs/NAZWA.json prepare static candidates     # lokalnie
    #   ... ocena kandydatów przez Suryę na GPU (polecenie wypisuje etap `candidates`) ...
    python rebuild.py configs/NAZWA.json rescore bundle ablations report

Opis etapów, pól konfiguracji i czasów: README.md w tym katalogu. `--dry-run` tylko wypisuje polecenia.

Gdzie co ląduje:
    ../results/NAZWA/           same liczby (do repo)
    data/NAZWA/groups.csv       linia -> grupa, split (do repo, bez tekstu)
    $PP_CACHE/runs/NAZWA/       pliki z tekstem linii (predykcje, kandydaci, logi); NIGDY do repo
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ocr_postprocess.bundle import MODELS_DIR
from ocr_postprocess.resources import cache_dir

HERE = Path(__file__).resolve().parent
STAGES = ("prepare", "static", "candidates", "rescore", "bundle", "ablations", "report")
REQUIRED = ("name", "benchmark_run", "dataset_dir", "dataset_id", "model", "images_prefix", "bundle_name")
DEFAULTS = {"compare_models": [], "group_rules": "data/group_rules.json", "domain": "en,la,pl", "harm_max": 0.003,
            "adapter": "<katalog adaptera LoRA>"}
ABLATIONS = {  # tag -> dodatkowe argumenty run_cv (baza: wariant główny)
    "main": [], "uniform": ["--uniform"], "no_multi": ["--no-multi"], "no_group_cap": ["--no-group-cap"],
    "no_protect": ["--no-protect"], "sjp": ["--sjp"], "forward": ["--forward"], "diacritics_only": ["--diacritics-only"],
    "no_domain": ["--domain", ""], "domain_la_pl": ["--domain", "la,pl"], "domain_in_tuning": ["--domain-in-tuning"],
    "harm_05": ["--harm-max", "0.005"], "harm_10": ["--harm-max", "0.01"],
}


class Run:
    def __init__(self, cfg: dict, dry: bool, force: bool):
        missing = [k for k in REQUIRED if k not in cfg]
        unknown = [k for k in cfg if k not in (*REQUIRED, *DEFAULTS) and not k.startswith("_")]
        if missing or unknown:
            raise SystemExit(f"konfiguracja: brakuje {missing}, nieznane {unknown}")
        self.c = {**DEFAULTS, **cfg}
        self.dry, self.force = dry, force
        name = self.c["name"]
        self.work = cache_dir() / "runs" / name
        self.results = HERE.parent / "results" / name
        self.groups = HERE / "data" / name / "groups.csv"
        self.preds = self.work / "preds"
        self.main_preds = self.preds / f"{self.c['model']}.jsonl"
        self.lp = self.work / "lp"

    def sh(self, module: str, *args, log: str) -> None:
        cmd = [sys.executable, "-m", f"evaluation.{module}", *map(str, args)]
        print("$ python -m", f"evaluation.{module}", " ".join(map(str, args)), flush=True)
        if self.dry:
            return
        logfile = self.work / "logs" / f"{log}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(logfile, "w", encoding="utf-8") as f:
            code = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            print(logfile.read_text(encoding="utf-8")[-3000:], file=sys.stderr)
            raise SystemExit(f"{module} zakończył się kodem {code}; log: {logfile}")
        print(f"  ok ({time.time() - t0:.0f} s), log: {logfile}", flush=True)

    def need(self, path: Path, why: str) -> None:
        if not self.dry and not path.exists():
            raise SystemExit(f"brak {path}: {why}")

    def fresh(self, *paths: Path) -> None:
        """Etap liczy od zera: stare pliki usuwa tylko z --force (p4_summary.csv dopisuje wiersze)."""
        old = [p for p in paths if p.exists()]
        if old and not self.force and not self.dry:
            raise SystemExit(f"wyniki już istnieją: {[str(p) for p in old]}; użyj --force albo innej nazwy przebiegu")
        for p in old:
            if not self.dry:
                shutil.rmtree(p) if p.is_dir() else p.unlink()

    def common(self, preds: Path) -> list:
        return ["--predictions", preds.resolve(), "--groups", self.groups]

    # ------------------------------------------------------------------ etapy
    def prepare(self) -> None:
        c = self.c
        self.fresh(self.preds, self.groups, self.results / "raw_ocr.csv")
        self.sh("import_bench", "--run", Path(c["benchmark_run"]).expanduser(),
                "--dataset-dir", Path(c["dataset_dir"]).expanduser(), "--dataset-id", c["dataset_id"],
                "--group-rules", (HERE / c["group_rules"]).resolve(), "--out", self.preds,
                "--groups-out", self.groups, log="prepare_import")
        for m in [c["model"], *c["compare_models"]]:
            self.need(self.preds / f"{m}.jsonl", f"w przebiegu benchmarku nie ma modelu {m}")
        self.sh("raw_ocr", "--preds", self.preds, "--groups", self.groups, "--out", self.results / "raw_ocr.csv",
                log="prepare_raw")
        if not self.dry:
            print((self.work / "logs" / "prepare_import.log").read_text(encoding="utf-8"))
            print((self.work / "logs" / "prepare_raw.log").read_text(encoding="utf-8"))
            print("SPRAWDŹ przed dalszymi etapami: grupy i podział dev/test powyżej (grupa = osobne źródło/autor).")

    def static(self) -> None:
        c = self.c
        self.need(self.groups, "najpierw etap prepare")
        self.fresh(self.results / "p4_summary.csv")
        for m in [c["model"], *c["compare_models"]]:
            args = [*self.common(self.preds / f"{m}.jsonl"), "--harm-max", c["harm_max"], "--domain", c["domain"],
                    "--tag", m, "--out-dir", self.results]
            self.sh("run_cv", *args, log=f"static_logo_{m}")
            self.sh("run_holdout", *args, "--test", log=f"static_holdout_{m}")
        self.sh("run_cv", *self.common(self.main_preds), "--harm-max", c["harm_max"], "--deploy",
                "--out-dir", self.results, log="static_deploy")

    def candidates(self) -> None:
        c = self.c
        self.need(self.main_preds, "najpierw etap prepare")
        self.fresh(self.lp)
        self.sh("rescore_cands", *self.common(self.main_preds), "--images-prefix", c["images_prefix"],
                "--out", self.lp, log="candidates")
        print(f"""
KROK NA GPU (ocena kandydatów przez Suryę, na RTX 3090 ok. 100 tekstów/s):
  1. skopiuj {self.lp / 'req.jsonl'} na maszynę z GPU (zawiera tekst linii);
  2. w repo, w kontenerze surya-training, z adapterem {c['adapter']}:
     docker compose -f training/docker-compose.yml run --rm -e PYTHONPATH=/app/postprocessing/package/src surya-training \\
         python postprocessing/experiments/evaluation/surya_lp.py --adapter {c['adapter']} --inp req.jsonl --out resC.jsonl
     (ścieżki req/resC względem /app; obrazy linii muszą leżeć w {c['images_prefix']})
  3. skopiuj resC.jsonl do {self.lp / 'resC.jsonl'}
Potem: python rebuild.py CONFIG rescore bundle ablations report
""")

    def rescore(self) -> None:
        c = self.c
        self.need(self.lp / "resC.jsonl", "wynik kroku na GPU (etap candidates wypisuje polecenie)")
        base = [self.lp, *self.common(self.main_preds), "--harm", c["harm_max"], "--out-dir", self.results]
        self.sh("run_rescore", *base, "--fixed-w", "0", log="rescore_w0")
        self.sh("run_rescore", *base, log="rescore_surya")
        self.sh("run_rescore", *base, "--fixed-w", "0", "--deploy", self.results / "deploy_params_top5.json",
                log="rescore_deploy_top5")
        self.sh("run_rescore", *base, "--deploy", self.results / "deploy_params_surya.json", log="rescore_deploy_surya")

    def bundle(self) -> None:
        c = self.c
        top5, surya = self.results / "deploy_params_top5.json", self.results / "deploy_params_surya.json"
        self.need(top5, "najpierw etap rescore")
        self.need(surya, "najpierw etap rescore")
        out = MODELS_DIR / c["bundle_name"]
        self.fresh(out)
        self.sh("build_bundle", *self.common(self.main_preds), "--params", top5, "--params-rescored", surya,
                "--domain", c["domain"], "--out", out, "--note",
                f"{c['model']}, przebieg {c['name']}; progi LOGO, harm<={c['harm_max']}", log="bundle")
        print(f"paczka: {out}\nŻeby była domyślna: DEFAULT_BUNDLE = {c['bundle_name']!r} w ocr_postprocess/bundle.py")

    def ablations(self) -> None:
        c = self.c
        self.need(self.main_preds, "najpierw etap prepare")
        out = self.results / "ablations"
        self.fresh(out)
        for report in ("oracle", "p2_report", "p3_report"):
            self.sh(report, *self.common(self.main_preds), "--out-dir", out, log=f"abl_{report}")
        for tag, extra in ABLATIONS.items():
            # późniejszy argument wygrywa w argparse, więc `extra` nadpisuje domenę i limit harm
            self.sh("run_cv", *self.common(self.main_preds), "--harm-max", c["harm_max"], "--domain", c["domain"],
                    "--tag", tag, "--out-dir", out, *extra, log=f"abl_{tag}")

    def report(self) -> None:
        if not self.dry:
            self.results.mkdir(parents=True, exist_ok=True)
            (self.results / "config.json").write_text(json.dumps(self.c, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8")
        self.sh("report", self.results, log="report")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("stages", nargs="+", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="nadpisz wyniki etapu z poprzedniego przebiegu")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    run = Run(cfg, a.dry_run, a.force)
    for s in a.stages:
        print(f"== {s} ==", flush=True)
        getattr(run, s)()


if __name__ == "__main__":
    main()
