"""Pack a directory of many small files into a few tar shards for DVC.

DVC hashes and uploads every file separately, so a directory of 800k JPEGs
takes hours to add and push. One giant tar fixes that but hits the remote's
per-request size limit (DagsHub answers 413 for a 13.5 GB body). Shards are
the middle ground: few enough files for DVC to stay fast, small enough for
the upload to go through.

Each shard is an independent tar — extracting all of them into one directory
reproduces the original, and no shard depends on its neighbours.

Usage:
    python src/shard_for_dvc.py output/20k_records/images \\
        --output-dir "C:/.../dataset/docs_20k/images_shards" --shard-gb 1
"""

import argparse
import sys
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack a directory into independently extractable tar shards."
    )
    parser.add_argument("source", type=str, help="Directory to pack.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where the shards are written.")
    parser.add_argument("--shard-gb", type=float, default=1.0,
                        help="Target size per shard in GB (default: 1.0).")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Shard file name prefix (default: source dir name).")
    return parser.parse_args()


def plan_shards(files: list[Path], limit_bytes: int) -> list[list[Path]]:
    """Group files into shards, each staying under the size limit.

    A single file larger than the limit still gets its own shard — splitting
    inside a file is not this tool's job, and the caller needs to know.
    """
    shards: list[list[Path]] = []
    current: list[Path] = []
    running = 0
    for path in files:
        size = path.stat().st_size
        if current and running + size > limit_bytes:
            shards.append(current)
            current, running = [], 0
        current.append(path)
        running += size
    if current:
        shards.append(current)
    return shards


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: not a directory: {source}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or source.name
    limit = int(args.shard_gb * 1024 ** 3)

    files = sorted(p for p in source.rglob("*") if p.is_file())
    if not files:
        print(f"ERROR: no files under {source}", file=sys.stderr)
        sys.exit(1)

    total = sum(p.stat().st_size for p in files)
    shards = plan_shards(files, limit)
    print(f"{len(files):,} plikow, {total / 1024**3:.2f} GB "
          f"-> {len(shards)} kawalkow po max {args.shard_gb} GB")

    oversized = [s[0] for s in shards if len(s) == 1 and s[0].stat().st_size > limit]
    for path in oversized:
        print(f"  UWAGA: {path.name} sam w sobie przekracza limit "
              f"({path.stat().st_size / 1024**3:.2f} GB)", file=sys.stderr)

    for index, shard in enumerate(shards):
        name = f"{prefix}_{index:03d}.tar"
        written = 0
        # Files are stored relative to the source directory, so extracting
        # every shard into one folder rebuilds the original layout.
        with tarfile.open(output_dir / name, "w") as archive:
            for path in shard:
                archive.add(path, arcname=str(path.relative_to(source)))
                written += path.stat().st_size
        print(f"  {name}  {len(shard):>7,} plikow  {written / 1024**3:6.2f} GB")

    print(f"\nGotowe: {output_dir}")
    print("Rozpakowanie u odbiorcy (kazdy kawalek do tego samego folderu):")
    print(f"  for f in {prefix}_*.tar; do tar -xf \"$f\" -C images/; done")


if __name__ == "__main__":
    main()
