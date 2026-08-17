"""Konwersja danych treningowych do formatu Surya OCR.

Surya (w fine-tuningu) oczekuje danych jako: obrazy (wycinki linii) + transkrypcje.
Kanonicalny format na wejściu do `cli.py train` to:
    processed/surya/{train,val}/metadata.jsonl   # {"file_name": ..., "text": ...}
    processed/surya/{train,val}/images/           # płaskie pliki obrazów

Ten skrypt obsługuje dwa źródła danych (odkryte na DagsHub, gałąź `dvc`):

1. `ocr800k`     — syntetyczne linie od Tomka (800k próbek)
   ├── images_shards/images_NNN.tar   (shardy, w środku płaskie `NNNNNNN.jpg`)
   ├── labels.jsonl                   {"file_name":"images/NNN.jpg","text":"...",
   │                                   "split":"train"|"val", "font":..., "size":[w,h], ...}
   ├── labels.txt                     TSV: images/NNN.jpg<TAB>text   (wszystkie)
   ├── labels_train.txt               TSV: tylko train
   └── labels_val.txt                 TSV: tylko val

2. `handlabeled` — ręcznie olabelowane linie (testset do benchmarku)
   ├── annotations.csv                filename,label,x_min,y_min,x_max,y_max,crop
   └── <page>/lines/lineN.jpg         wycinki

Użycie:
    python convert_data.py ocr800k --input dataset/ocr_800k --output data/processed/surya
    python convert_data.py handlabeled --input dataset --output data/processed/surya
    python convert_data.py ocr800k --input dataset/ocr_800k --output ... --max-samples 1000
"""

import argparse
import json
import shutil
import tarfile
from pathlib import Path


# ---------------------------------------------------------------------------
# ocr800k → format Surya
# ---------------------------------------------------------------------------

def _load_ocr800k_labels(labels_path: Path) -> list[dict]:
    """Wczytuje labels.jsonl lub labels.txt (TSV) do listy {file_name, text, split}.

    UWAGA: labels mają ścieżkę "images/NNNNNNN.jpg", ale w tar-ach pliki są
    płaskie ("NNNNNNN.jpg"). Normalizujemy do samego basename, żeby dopasować
    wpisy labels do memberów tar-ów (i do płaskiego images/ na wyjściu).
    """
    samples = []
    if labels_path.suffix == ".jsonl":
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                text = (e.get("text") or "").strip()
                if not text:
                    continue
                samples.append({
                    "file_name": Path(e["file_name"]).name,  # "NNNNNNN.jpg"
                    "text": text,
                    "split": e.get("split", "train"),
                })
    else:  # .txt — TSV: images/NNN.jpg<TAB>text
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                file_name, text = parts
                text = text.strip()
                if not text:
                    continue
                samples.append({
                    "file_name": Path(file_name).name,  # "NNNNNNN.jpg"
                    "text": text,
                    "split": "train",
                })
    return samples


def _stratified_sample(samples: list[dict], max_samples: int) -> list[dict]:
    """Ogranicza liczbę próbek zachowując proporcję train/val.

    Bez tego `samples[:max_samples]` brałoby wyłącznie train (labels.jsonl ma
    najpierw wszystkie train), a val wyszłoby puste i `cli.py train` padłby
    przy ewaluacji.
    """
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] != "train"]
    total = len(samples) or 1
    n_train = max(1, int(max_samples * len(train) / total))
    n_val = max_samples - n_train
    n_train = min(n_train, len(train))
    n_val = min(n_val, len(val))
    return train[:n_train] + val[:n_val]


def convert_ocr800k(input_dir: Path, output_dir: Path, max_samples: int | None) -> None:
    """Konwertuje ocr_800k (tar + labels) do formatu Surya.

    Strategia: jednoprzebiegowa ekstrakcja tar-ów z routingiem do train/val
    (bez podwójnego kopiowania 800k obrazów).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 1. labels — preferuj labels.jsonl (ma pole split); fallback labels.txt
    jsonl = input_dir / "labels.jsonl"
    txt = input_dir / "labels.txt"
    labels_path = jsonl if jsonl.exists() else txt
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Brak labels.jsonl ani labels.txt w {input_dir}"
        )

    samples = _load_ocr800k_labels(labels_path)
    if max_samples:
        samples = _stratified_sample(samples, max_samples)

    # file_name -> (split, text)
    split_map = {s["file_name"]: s["split"] for s in samples}
    text_map = {s["file_name"]: s["text"] for s in samples}

    # 2. shardy
    shards_dir = input_dir / "images_shards"
    shards = sorted(shards_dir.glob("images_*.tar"))
    if not shards:
        raise FileNotFoundError(f"Brak shardów images_*.tar w {shards_dir}")

    # 3. przygotuj katalogi wyjściowe
    train_meta, val_meta = [], []
    n_extracted = 0

    for shard in shards:
        with tarfile.open(shard) as tf:
            try:
                members = tf.getmembers()
            except tarfile.ReadError as exc:
                print(f"Pominięto niekompletny shard {shard.name}: {exc}")
                continue
            for member in members:
                if not member.isfile():
                    continue
                name = member.name                      # płaskie "NNNNNNN.jpg"
                if name not in split_map:
                    continue
                split = split_map[name]
                dest = output_dir / split / "images" / Path(name).name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                n_extracted += 1
                meta = {"file_name": Path(name).name, "text": text_map[name]}
                (train_meta if split == "train" else val_meta).append(meta)

    # 4. zapisz metadata.jsonl
    for split, meta in [("train", train_meta), ("val", val_meta)]:
        out_meta = output_dir / split / "metadata.jsonl"
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as f:
            for line in meta:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{split}: {len(meta)} próbek → {out_meta}")

    print(f"Wyekstrahowano {n_extracted} obrazów z {len(shards)} shardów.")


# ---------------------------------------------------------------------------
# handlabeled → format Surya
# ---------------------------------------------------------------------------

def convert_handlabeled(input_dir: Path, output_dir: Path,
                        val_split: float) -> None:
    """Konwertuje ręcznie olabelowane wycinki (annotations.csv + lines/) → Surya.

    Wejściowy annotations.csv: filename,label,x_min,y_min,x_max,y_max,crop
    gdzie `crop` to nazwa pliku wycinka w <page>/lines/.
    """
    import csv

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    csv_path = input_dir / "annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Brak annotations.csv w {input_dir}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("label") or "").strip()
            page = (row.get("filename") or "").strip()
            crop = (row.get("crop") or "").strip()
            if not (text and page and crop):
                continue
            # filename w CSV to nazwa oryginalnego obrazu (np. "data11.png"),
            # a katalog strony to ta nazwa bez rozszerzenia ("data11/").
            page_dir = Path(page).stem
            samples.append({"page": page_dir, "crop": crop, "text": text})

    # Podział train/val (deterministyczny: kolejność z CSV)
    split_idx = int(len(samples) * (1 - val_split))
    splits = {"train": samples[:split_idx], "val": samples[split_idx:]}

    for split, ss in splits.items():
        images_dir = output_dir / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        meta = []
        for s in ss:
            src = input_dir / s["page"] / "lines" / s["crop"]
            if not src.exists():
                continue
            # unikalna nazwa: <page>_<crop>
            flat_name = f"{s['page']}_{s['crop']}"
            dst = images_dir / flat_name
            if not dst.exists():
                shutil.copy2(src, dst)
            meta.append({"file_name": flat_name, "text": s["text"]})
        out_meta = output_dir / split / "metadata.jsonl"
        with open(out_meta, "w", encoding="utf-8") as f:
            for line in meta:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{split}: {len(meta)} próbek → {out_meta}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="source", required=True)

    # ocr800k
    o = sub.add_parser("ocr800k", help="Syntetyczne linie (tar + labels.jsonl)")
    o.add_argument("--input", required=True, help="Katalog ocr_800k/")
    o.add_argument("--output", required=True, help="Katalog wyjściowy (processed/surya/)")
    o.add_argument("--max-samples", type=int, default=None,
                   help="Ogranicz liczbę próbek (warstwowo, do testów)")
    o.set_defaults(fn=lambda a: convert_ocr800k(
        Path(a.input), Path(a.output), a.max_samples))

    # handlabeled
    h = sub.add_parser("handlabeled", help="Ręcznie olabelowane wycinki")
    h.add_argument("--input", required=True, help="Katalog z annotations.csv")
    h.add_argument("--output", required=True, help="Katalog wyjściowy")
    h.add_argument("--val-split", type=float, default=0.1)
    h.set_defaults(fn=lambda a: convert_handlabeled(
        Path(a.input), Path(a.output), a.val_split))

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
