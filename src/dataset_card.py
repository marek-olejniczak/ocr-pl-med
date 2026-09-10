"""Provenance for generated datasets.

Every run drops a dataset_card.json next to the images and adds a row to a
registry table in the repo, so months later it is still possible to answer
"what exactly was in that dataset and how was it made?".

The augmentation section is dumped straight from the code's own constants —
nothing is retyped by hand, so the card cannot drift away from what the
generator actually did.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import artifacts
import fill_form
import field_content

REGISTRY_HEADER = "# Wygenerowane zbiory danych\n"
REGISTRY_INTRO = (
    "\nJeden wiersz na zbiór. Karta z pełnymi ustawieniami leży w folderze\n"
    "zbioru jako `dataset_card.json` (foldery z danymi nie są w repo).\n\n"
)
_TABLE_HEAD = (
    "| Data | Nazwa | Commit | Seed | Obrazy | Anotacje | Notatka |\n"
    "|---|---|---|---|---|---|---|\n"
)


def augmentation_settings() -> dict:
    """Dump every augmentation knob the generator currently uses.

    `generate_yolo_dataset` is imported lazily: it imports this module to
    write the card, so importing it at module level would be circular.
    """
    import generate_yolo_dataset

    return {
        "stamp_prob": artifacts.STAMP_PROB,
        "stamp_round_prob": artifacts.STAMP_ROUND_PROB,
        "stamp_angle_max": artifacts.STAMP_ANGLE_MAX,
        "stamp_gt_max_angle": artifacts.STAMP_GT_MAX_ANGLE,
        "stroke_prob": artifacts.STROKE_PROB,
        "highlight_prob": artifacts.HIGHLIGHT_PROB,
        "highlight_min_luma": artifacts.HIGHLIGHT_MIN_LUMA,
        "pen_fade_prob": fill_form.PEN_FADE_PROB,
        "v_overflow_frac": fill_form.V_OVERFLOW_FRAC,
        "form_font_size_range": list(fill_form.FORM_FONT_SIZE_RANGE),
        "min_font_size": fill_form.MIN_FONT_SIZE,
        "max_font_size": fill_form.MAX_FONT_SIZE,
        "multiline_max_lines": fill_form.MULTILINE_MAX_LINES,
        "line_pitch_range": list(fill_form.LINE_PITCH_RANGE),
        "scan_profiles": [[name, weight] for name, weight in fill_form.SCAN_PROFILES],
        "medical_text_prob": field_content.MEDICAL_TEXT_PROB,
        "fill_frac_range": [field_content.FILL_FRAC_MIN, field_content.FILL_FRAC_MAX],
        "partial_base_prob": generate_yolo_dataset.PARTIAL_BASE_PROB,
    }


def git_info(repo_root: Path) -> dict:
    """Short commit hash plus whether the tree had uncommitted changes.

    Without the dirty flag the commit hash can quietly lie about what code
    produced the data.
    """
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(repo_root),
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "--short", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": False}
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def build_card(
    name: str,
    command: str,
    seed: Optional[int],
    repo_root: Path,
    counts: dict,
    sources: dict,
    fonts: list[str],
    observed: dict,
    note: str = "",
) -> dict:
    """Assemble the full provenance record for one generation run."""
    info = git_info(repo_root)
    return {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": info["commit"],
        "git_dirty": info["dirty"],
        "command": command,
        "seed": seed,
        "counts": counts,
        "sources": sources,
        "fonts": sorted(fonts),
        "augmentations": augmentation_settings(),
        "observed": observed,
        "note": note,
    }


def write_card(output_dir: Path, card: dict) -> Path:
    """Write dataset_card.json into the dataset directory."""
    path = Path(output_dir) / "dataset_card.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(card, handle, ensure_ascii=False, indent=2)
    return path


def _row(card: dict) -> str:
    # Document datasets count images/annotations, line datasets count
    # lines/train — the table shows whichever the card carries.
    counts = card["counts"]
    return (
        f"| {card['created'][:10]} | {card['name']} | "
        f"{card['git_commit'] or '-'}{'*' if card['git_dirty'] else ''} | "
        f"{card['seed'] if card['seed'] is not None else '-'} | "
        f"{counts.get('images', counts.get('lines', '-'))} | "
        f"{counts.get('annotations', counts.get('train', '-'))} | {card.get('note', '')} |\n"
    )


def update_registry(registry_path: Path, card: dict) -> None:
    """Add or replace this dataset's row in the markdown registry."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            if stripped.startswith("| Data |") or set(stripped) <= set("|- "):
                continue
            rows.append(line)

    key = f"| {card['name']} |"
    rows = [r for r in rows if key not in r]
    rows.append(_row(card))
    rows.sort()

    path.write_text(
        REGISTRY_HEADER + REGISTRY_INTRO + _TABLE_HEAD + "".join(rows),
        encoding="utf-8",
    )
