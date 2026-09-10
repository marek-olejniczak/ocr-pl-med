"""Generate and pack the OCR ablation datasets in one go.

Five datasets, one command each way:

    v2_800k     everything on, full size          -> the actual model (phase 2)
    v2_200k     everything on, quarter size       -> ablation reference
    noA_200k    without content families          -> "does WHAT is written matter"
    noB_200k    without framing / background      -> "does WHERE it is written matter"
    noC_200k    without image degradation         -> "does HOW it was captured matter"

The four 200k sets share one seed, so they differ only by the disabled group.
Generation is resumable: a variant whose output already has a dataset_card.json
is skipped, so a crashed run can simply be started again.

Usage:
    python src/build_ablation_sets.py generate                 # all five, in order
    python src/build_ablation_sets.py generate --only noA_200k
    python src/build_ablation_sets.py pack --dvc-repo "C:/Users/tomek/Desktop/dane/inzynierka"
    python src/build_ablation_sets.py list
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GROUP_A = "short_words,caps,arrows_bullets,anatomy_vocab"
GROUP_B = "neighbour_glyphs,grid_paper"
GROUP_C = "morphology,elastic,phone_photo"

# name -> (count, seed, --disable value, note)
VARIANTS: dict[str, tuple[int, int, str, str]] = {
    "v2_800k": (800_000, 2029, "",
                "generator v2, wszystko wlaczone; model wlasciwy (faza 2)"),
    "v2_200k": (200_000, 2030, "",
                "ablacja: punkt odniesienia, wszystko wlaczone"),
    "noA_200k": (200_000, 2030, GROUP_A,
                 "ablacja: bez grupy A (tresc: short_words, caps, arrows_bullets, anatomy_vocab)"),
    "noB_200k": (200_000, 2030, GROUP_B,
                 "ablacja: bez grupy B (kadr i tlo: neighbour_glyphs, grid_paper)"),
    "noC_200k": (200_000, 2030, GROUP_C,
                 "ablacja: bez grupy C (degradacja: morphology, elastic, phone_photo)"),
}
# Files convert_data.py (branch `server`) reads next to images_shards/.
LABEL_FILES = ("labels.jsonl", "labels.txt", "labels_train.txt", "labels_val.txt", "dataset_card.json")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Render the datasets with generate_ocr_lines.py")
    g.add_argument("--output-root", default="output", help="Where output/<name>/ goes")
    g.add_argument("--only", nargs="*", default=None, help="Subset of variant names")
    g.add_argument("--scale", type=float, default=1.0,
                   help="Multiply every count (0.001 for a smoke test)")
    g.add_argument("--registry", default=None,
                   help="Passed through to the generator ('none' for smoke tests)")

    p = sub.add_parser("pack", help="Shard images and copy labels into the DVC repo")
    p.add_argument("--output-root", default="output")
    p.add_argument("--dvc-repo", required=True, help="Clone of the DagsHub repo")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--shard-gb", type=float, default=1.0)

    sub.add_parser("list", help="Print the variant table")
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def selected(only) -> list[str]:
    names = list(VARIANTS) if not only else list(only)
    unknown = set(names) - set(VARIANTS)
    if unknown:
        sys.exit(f"unknown variants: {sorted(unknown)}; choose from {list(VARIANTS)}")
    return names


def cmd_generate(args: argparse.Namespace) -> None:
    for name in selected(args.only):
        count, seed, disable, note = VARIANTS[name]
        out = Path(args.output_root) / name
        if (out / "dataset_card.json").exists():
            print(f"[{name}] already complete, skipping")
            continue
        cmd = [sys.executable, "src/generate_ocr_lines.py",
               "--output-dir", str(out),
               "--count", str(max(1, int(count * args.scale))),
               "--seed", str(seed),
               "--dataset-name", name,
               "--note", note]
        if disable:
            cmd += ["--disable", disable]
        if args.registry:
            cmd += ["--registry", args.registry]
        run(cmd)


def cmd_pack(args: argparse.Namespace) -> None:
    dvc_repo = Path(args.dvc_repo)
    for name in selected(args.only):
        src = Path(args.output_root) / name
        if not (src / "dataset_card.json").exists():
            print(f"[{name}] not generated yet, skipping")
            continue
        dst = dvc_repo / "dataset" / name
        dst.mkdir(parents=True, exist_ok=True)
        # convert_data.py expects images_shards/images_NNN.tar with flat jpgs
        # inside, plus labels.jsonl next to it.
        run([sys.executable, "src/shard_for_dvc.py", str(src / "images"),
             "--output-dir", str(dst / "images_shards"),
             "--prefix", "images", "--shard-gb", str(args.shard_gb)])
        for fname in LABEL_FILES:
            if (src / fname).exists():
                shutil.copy2(src / fname, dst / fname)
        print(f"[{name}] packed into {dst}")
    print("\nNext, inside the DVC repo:")
    print("  dvc add dataset && git add dataset.dvc && git commit -m 'ablation sets' && dvc push && git push")


def cmd_list() -> None:
    print(f"{'name':10} {'count':>8} {'seed':>5}  disable")
    for name, (count, seed, disable, _) in VARIANTS.items():
        print(f"{name:10} {count:>8} {seed:>5}  {disable or '-'}")


def main() -> None:
    args = parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "pack":
        cmd_pack(args)
    else:
        cmd_list()


if __name__ == "__main__":
    main()
