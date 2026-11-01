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
    python src/build_ablation_sets.py generate --only rodziny   # the 9 per-family sets
    python src/build_ablation_sets.py generate --only v3        # generator v3 pilot (5 sets)
    python src/build_ablation_sets.py pack --dvc-repo "C:/Users/tomek/Desktop/dane/inzynierka"
    python src/build_ablation_sets.py list
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_card import update_registry, write_card  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

GROUP_A = "short_words,caps,arrows_bullets,anatomy_vocab"
GROUP_B = "neighbour_glyphs,grid_paper"
GROUP_C = "morphology,elastic,phone_photo"
# Generator v3 glyph families. Off in every v2 variant, so those keep
# reproducing the datasets that were trained on.
GLYPHS = "cursive_joins,glyph_variants,diacritic_placement"


def _v2(disable: str) -> str:
    return ",".join(x for x in (disable, GLYPHS) if x)

# name -> (count, seed, --disable value, note)
VARIANTS: dict[str, tuple[int, int, str, str]] = {
    "v2_800k": (800_000, 2029, _v2(""),
                "generator v2, wszystko wlaczone; model wlasciwy (faza 2)"),
    "v2_200k": (200_000, 2030, _v2(""),
                "ablacja: punkt odniesienia, wszystko wlaczone"),
    "noA_200k": (200_000, 2030, _v2(GROUP_A),
                 "ablacja: bez grupy A (tresc: short_words, caps, arrows_bullets, anatomy_vocab)"),
    "noB_200k": (200_000, 2030, _v2(GROUP_B),
                 "ablacja: bez grupy B (kadr i tlo: neighbour_glyphs, grid_paper)"),
    "noC_200k": (200_000, 2030, _v2(GROUP_C),
                 "ablacja: bez grupy C (degradacja: morphology, elastic, phone_photo)"),
}

# Leave-one-out per family (second ablation round, 2026-10). Same seed and
# size as v2_200k, which stays their reference: the generator change that
# added parameter logging reproduces v2_200k's text and metadata exactly.
FAMILIES = ("short_words", "caps", "arrows_bullets", "anatomy_vocab",
            "neighbour_glyphs", "grid_paper", "morphology", "elastic", "phone_photo")
for _fam in FAMILIES:
    VARIANTS[f"no_{_fam}_200k"] = (200_000, 2030, _v2(_fam), f"ablacja rodzin: bez {_fam}")

# Generator v3 pilot (2026-11): the v2 generator plus the glyph families, one
# set without each of them, and v3_base without all three. All five are
# rendered with --line-seeds (every line and every stage of it reseeded), so
# they share text, fonts, split and every other effect line for line and
# differ only in the glyph families. v3_base is the reference; it is v2 in
# content, but not v2_200k's lines.
GLYPH_FAMILIES = tuple(GLYPHS.split(","))
LINE_SEEDS = {"v3_base_200k", "v3_200k"}
VARIANTS["v3_base_200k"] = (200_000, 2031, GLYPHS,
                            "generator v3 pilot: odniesienie, v2 bez rodzin glifow (line seeds)")
VARIANTS["v3_200k"] = (200_000, 2031, "",
                       "generator v3 pilot: v2 + cursive_joins, glyph_variants, diacritic_placement")
for _fam in GLYPH_FAMILIES:
    VARIANTS[f"v3_no_{_fam}_200k"] = (200_000, 2031, _fam, f"generator v3 pilot: bez {_fam}")
    LINE_SEEDS.add(f"v3_no_{_fam}_200k")
# --only accepts these names as shorthands for a whole round.
SETS = {
    "grupy": ["v2_800k", "v2_200k", "noA_200k", "noB_200k", "noC_200k"],
    "rodziny": [f"no_{f}_200k" for f in FAMILIES],
    "v3": ["v3_base_200k", "v3_200k"] + [f"v3_no_{f}_200k" for f in GLYPH_FAMILIES],
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
    g.add_argument("--jobs", type=int, default=1,
                   help="Render a line-seeded set in this many parallel pieces "
                        "and join them; the result is the same as one run.")

    p = sub.add_parser("pack", help="Shard images and copy labels into the DVC repo")
    p.add_argument("--output-root", default="output")
    p.add_argument("--dvc-repo", required=True, help="Clone of the DagsHub repo")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--shard-gb", type=float, default=0.9,
                   help="Max tar size per shard in GiB; DagsHub cuts files over 1 GiB.")

    sub.add_parser("list", help="Print the variant table")
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def selected(only) -> list[str]:
    names = list(VARIANTS) if not only else [n for o in only for n in SETS.get(o, [o])]
    unknown = set(names) - set(VARIANTS)
    if unknown:
        sys.exit(f"unknown variants: {sorted(unknown)}; choose from {list(VARIANTS)}")
    return names


def generator_cmd(name: str, out: Path, count: int, registry) -> list[str]:
    _, seed, disable, note = VARIANTS[name]
    cmd = [sys.executable, "src/generate_ocr_lines.py",
           "--output-dir", str(out),
           "--count", str(count),
           "--seed", str(seed),
           "--dataset-name", name,
           "--note", note]
    if disable:
        cmd += ["--disable", disable]
    if name in LINE_SEEDS:
        cmd += ["--line-seeds"]
    if registry:
        cmd += ["--registry", registry]
    return cmd


LABEL_PARTS = ("labels.txt", "labels_train.txt", "labels_val.txt", "labels.jsonl")


def generate_in_pieces(name: str, out: Path, count: int, jobs: int, registry) -> None:
    """Render a line-seeded set in parallel pieces, then join them into `out`."""
    parts_root = out / "_parts"
    size = -(-count // jobs)
    pieces, logs = [], []
    for k in range(jobs):
        start = k * size
        n = min(size, count - start)
        if n <= 0:
            break
        part = parts_root / f"part{k:02d}"
        if (part / "dataset_card.json").exists():
            print(f"[{name}] piece {k} already complete")
            pieces.append((part, None))
            continue
        cmd = generator_cmd(name, part, n, "none") + ["--start-index", str(start)]
        print("+", " ".join(cmd), flush=True)
        parts_root.mkdir(parents=True, exist_ok=True)
        log = open(parts_root / f"part{k:02d}.log", "w", encoding="utf-8")
        logs.append(log)
        pieces.append((part, subprocess.Popen(cmd, cwd=str(REPO), stdout=log,
                                              stderr=subprocess.STDOUT)))
    failed = [p for p, proc in pieces if proc is not None and proc.wait() != 0]
    for log in logs:
        log.close()
    if failed:
        sys.exit(f"[{name}] pieces failed: {failed}; rerun to redo only those")

    (out / "images").mkdir(parents=True, exist_ok=True)
    handles = {f: open(out / f, "w", encoding="utf-8") for f in LABEL_PARTS}
    cards = []
    for part, _ in pieces:
        for f in LABEL_PARTS:
            handles[f].write((part / f).read_text(encoding="utf-8"))
        for img in (part / "images").iterdir():
            img.replace(out / "images" / img.name)
        cards.append(json.loads((part / "dataset_card.json").read_text(encoding="utf-8")))
    for h in handles.values():
        h.close()

    card = cards[0]
    for key in ("lines", "train", "val"):
        card["counts"][key] = sum(c["counts"][key] for c in cards)
    card["sources"] = {"synthetic": card["counts"]["lines"]}
    card["fonts"] = sorted({f for c in cards for f in c["fonts"]})
    for key in ("scan_profiles", "content_kinds"):
        merged: dict = {}
        for c in cards:
            for k, v in c["observed"][key].items():
                merged[k] = merged.get(k, 0) + v
        card["observed"][key] = merged
    card["observed"]["empty_content_retries"] = sum(
        c["observed"]["empty_content_retries"] for c in cards)
    card["command"] = ("python " + " ".join(generator_cmd(name, out, count, registry)[1:])
                       + f"  (w {len(pieces)} kawalkach rownolegle, --start-index)")
    write_card(out, card)
    if registry != "none":
        update_registry(Path(registry) if registry else REPO / "docs" / "datasets.md", card)
    shutil.rmtree(parts_root)
    print(f"[{name}] joined {len(pieces)} pieces: {card['counts']}")


def cmd_generate(args: argparse.Namespace) -> None:
    for name in selected(args.only):
        count = max(1, int(VARIANTS[name][0] * args.scale))
        out = Path(args.output_root) / name
        if (out / "dataset_card.json").exists():
            print(f"[{name}] already complete, skipping")
            continue
        if args.jobs > 1 and name in LINE_SEEDS:
            generate_in_pieces(name, out, count, args.jobs, args.registry)
        else:
            run(generator_cmd(name, out, count, args.registry))


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
