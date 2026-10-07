"""Shards must stay under the size limit: DagsHub silently truncates files over 1 GiB."""

import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_shards_never_exceed_the_limit_and_keep_every_file(tmp_path):
    src = tmp_path / "images"
    src.mkdir()
    for i in range(400):
        (src / f"{i:07d}.jpg").write_bytes(bytes([i % 251]) * (3000 + 37 * i))
    out = tmp_path / "shards"
    limit_gb = 0.0005  # ~0.5 MiB, forces several shards
    result = subprocess.run(
        [sys.executable, str(REPO / "src" / "shard_for_dvc.py"), str(src),
         "--output-dir", str(out), "--prefix", "images", "--shard-gb", str(limit_gb)],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stderr

    limit = int(limit_gb * 1024 ** 3)
    shards = sorted(out.glob("images_*.tar"))
    assert len(shards) > 1
    names = []
    for shard in shards:
        assert shard.stat().st_size <= limit, shard.name
        with tarfile.open(shard) as tf:
            names += tf.getnames()
    assert sorted(names) == sorted(p.name for p in src.iterdir())
