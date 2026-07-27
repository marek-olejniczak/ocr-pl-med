import json
from pathlib import Path

from dataset_card import (
    augmentation_settings,
    git_info,
    build_card,
    write_card,
    update_registry,
)

REPO = Path(__file__).resolve().parent.parent


def test_settings_are_read_from_the_code():
    settings = augmentation_settings()
    assert settings["stamp_prob"] == 0.30
    assert settings["stroke_prob"] == 0.20
    assert settings["highlight_prob"] == 0.12
    assert settings["pen_fade_prob"] == 0.15
    assert settings["form_font_size_range"] == [26, 40]
    assert settings["stamp_gt_max_angle"] == 5.0
    assert settings["highlight_min_luma"] == 150.0
    assert ["clean_color", 0.45] in settings["scan_profiles"]


def test_settings_track_the_constants_rather_than_copying_them():
    """The card must not drift: it reads the live constant, not a literal."""
    import artifacts

    original = artifacts.STAMP_PROB
    try:
        artifacts.STAMP_PROB = 0.99
        assert augmentation_settings()["stamp_prob"] == 0.99
    finally:
        artifacts.STAMP_PROB = original
    assert augmentation_settings()["stamp_prob"] == original


def test_settings_are_json_serialisable():
    json.dumps(augmentation_settings())


def test_git_info_reads_the_repo():
    info = git_info(REPO)
    assert info["commit"] is None or len(info["commit"]) == 7
    assert isinstance(info["dirty"], bool)


def test_git_info_survives_a_non_repo(tmp_path):
    info = git_info(tmp_path)
    assert info == {"commit": None, "dirty": False}


def _card(name="demo", images=10, annotations=100):
    return build_card(
        name=name,
        command="python src/generate_yolo_dataset.py --output-dir output/demo",
        seed=42,
        repo_root=REPO,
        counts={"images": images, "annotations": annotations, "templates": 4},
        sources={"printed": 60, "synthetic": 30, "stamp": 10},
        fonts=["Marek_1-Regular.ttf"],
        observed={"scan_profiles": {"grayscale": 10}, "bases": {"blank": 10}},
        note="proba",
    )


def test_card_has_every_required_section():
    card = _card()
    for key in ("name", "created", "git_commit", "git_dirty", "command", "seed",
                "counts", "sources", "fonts", "augmentations", "observed", "note"):
        assert key in card, key
    assert card["augmentations"]["stamp_prob"] == 0.30


def test_write_card_lands_next_to_the_data(tmp_path):
    path = write_card(tmp_path, _card())
    assert path == tmp_path / "dataset_card.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["name"] == "demo"
    assert loaded["counts"]["images"] == 10


def test_registry_is_created_then_updated_in_place(tmp_path):
    registry = tmp_path / "datasets.md"

    update_registry(registry, _card(name="alpha", images=10))
    update_registry(registry, _card(name="beta", images=20))
    text = registry.read_text(encoding="utf-8")
    assert text.count("| alpha |") == 1
    assert text.count("| beta |") == 1

    # regenerating the same dataset replaces its row instead of duplicating
    update_registry(registry, _card(name="alpha", images=999))
    text = registry.read_text(encoding="utf-8")
    assert text.count("| alpha |") == 1
    assert "999" in text
    assert text.count("| beta |") == 1


def test_registry_keeps_the_table_readable(tmp_path):
    registry = tmp_path / "datasets.md"
    update_registry(registry, _card(name="alpha"))
    lines = registry.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#")
    header = next(i for i, l in enumerate(lines) if l.startswith("| Data |"))
    assert set(lines[header + 1]) <= set("|- ")  # separator row present
    assert lines[header + 2].startswith("| ")     # then the data row
