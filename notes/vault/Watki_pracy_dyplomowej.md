# Wątki do pracy dyplomowej

Zbiór wszystkich wątków/etapów/problematyk projektu `ocr-pl-med`, które nadają się do opisania
w pracy. Surowiec do ułożenia zarysu (luźna chronologia) → spisu treści. Stan na październik 2026;
postprocessing (wątek J, faza 7) dopisany 2026-11-26. Pozostałe wątki nie były wtedy aktualizowane.

## Status obecny

Projekt: **OCR/HTR polskich rękopiśmiennych dokumentów medycznych** (recepty, notatki, skany
podręczników anatomii). Pipeline: *preprocessing → detekcja/segmentacja linii pisma → per-line OCR*.

Dwa równoległe tory:

- **Detekcja/segmentacja linii — aktywny tor** (ostatnie commity, X 2026). HPO (sterownik Optuna),
  ablacja „round 2" (robustness + leave-one-family-out), lr sweep, r2/r3. Modele: YOLOv8/11,
  RT-DETR, Faster/Mask R-CNN (detectron2), DBNet (docTR), Kraken BLLA, Surya-det.
- **Rozpoznawanie tekstu — zakończone, czeka na tabelę porównawczą.** Fine-tuning: Surya (LoRA,
  ablacja datasetów rozstrzygnięta → `v2_200k`, CER 0.127243; sweep HP zamknięty bez zwycięzcy),
  PARSeq (docTR, 32×128 i 32×512 gotowe), TrOCR (pełny FT). Zero-shot baseline: ~12 modeli.

## Chronologia projektu (luźna oś do zarysu)

| Faza | Okres | Treść |
|---|---|---|
| 0 | I–III 2026 | setup, Obsidian vault, literatura, lista zadań, wybór zbiorów |
| 1 | III–IV 2026 | narzędzie etykietowania, usługi modeli (Docker/FastAPI), autorunner |
| 2 | V–VI 2026 | benchmark: metryki, evaluate, split datasetu, preprocessing |
| 3 | VI–VIII 2026 | detekcja linii: YOLO/RT-DETR/Faster R-CNN/Kraken, orchestrator, diagnostyka |
| 4 | VII 2026 | pipeline app (gradio) + e2e benchmark |
| 5 | VIII–IX 2026 | fine-tuning rozpoznawania (Surya LoRA, PARSeq, TrOCR), benchmark porównawczy |
| 6 | IX–X 2026 | Surya HP sweep, PARSeq wrapper benchmarku; HPO + ablacja detekcji linii |
| 7 | X–XI 2026 | postprocessing wyjścia OCR: korektor, ocena kandydatów przez Suryę, pakiet do aplikacji |

## Klasyfikacja wątków

11 obszarów tematycznych, wątki ponumerowane. Odnośniki wskazują, gdzie w repo jest materiał.

### A. Motywacja i kontekst dziedzinowy
- **A1** Problem nieczytelnego pisma w dokumentacji medycznej — błędy lekowe, bezpieczeństwo
  pacjenta, cyfryzacja (motywacja pracy).
- **A2** Specyfika polskiego OCR/HTR: diakrytyki (`ąćęłńóśźż`), znaki rzadkie, wyrazy spoza
  słownika (OOV).
- **A3** Specyfika domeny medycznej: terminologia, skróty lekarskie, nazwy leków (Rejestr
  Produktów Leczniczych).
- **A4** Ograniczenie RODO → brak dostępu do autentycznych recept → konieczność danych syntetycznych.

### B. Przegląd literatury i prac pokrewnych
Materiał: [[Prace_OCR]], [[Segmentacja_linii]].
- **B1** Architektury HTR: CNN+BiLSTM+CTC (Puigcerver), MDLSTM, ViT/seq2seq.
- **B2** Segmentacja linii: connected-components, GAN, DETR/LineTR, „struck-out text".
- **B3** Generowanie syntetyczne: HWT, ScrabbleGAN, GANwriting, InkSight.
- **B4** Adaptacja do autora: SEN, MetaHTR, MetaWriter (prompt tuning / test-time adaptation).
- **B5** Skalowalne HTR dla języka polskiego (PL-500k-synthetic, PL-30k-diacritics, PL-20k) —
  praca najbliżej analogiczna do naszej.

### C. Zbiory danych i ich przygotowanie
- **C1** Przegląd dostępnych zbiorów: RxHandBD, PHCD, PG-HWLD, RPL, korpusy medyczne
  ([[Zbiory_danych]]).
- **C2** Budowa własnego zbioru dokumentów medycznych (katalogi `dataset/`: atlasy anatomii,
  notatki, skany).
- **C3** Narzędzie etykietowania: floating label bar, `annotations.csv`, kolumna crop-filename.
- **C4** Wersjonowanie danych: DVC + DagsHub (`dataset.dvc`, `dvc pull`).
- **C5** Podział danych per strona-szablonu (`build_dataset.py --group`) — warianty jednej strony
  nie mogą trafić do różnych splitów.
- **C6** Zbiory ablacyjne: `v2`, `noA/noB/noC`, `leave-one-family-out`, `v2_val10k`, robustness.

### D. Metodyka benchmarkowania i metryki
- **D1** Harness: modele jako usługi Docker/FastAPI, `autorunner.py`, `experiments.yaml`
  (`benchmark/`).
- **D2** Metryki OCR: EMA, NPM (≤1 znak błędu), CER, WER, odległość Levenshteina
  (`benchmark/src/metrics.py`).
- **D3** Metryki detekcji linii: mAP, IoU, split/merge rate, ECE, błędy cy/ar, line-count
  accuracy (`line_benchmark/evaluation/metrics.py`).
- **D4** Metryki per-source / per-annotation (nie per strona).
- **D5** Timing i zasoby: czas inferencji per-obraz, peak GPU/RAM w `meta.json`.
- **D6** Ewaluacja na hoście vs w kontenerze; `--no-map` (COCO mAP ignoruje próg conf).

### E. Detekcja / segmentacja linii pisma
Materiał: `line_benchmark/`, [[Modele_segmentacja_linii]].
- **E1** Rodziny modeli i filozofie: YOLOv8/11, RT-DETR, Mask/Faster R-CNN, DBNet, Kraken BLLA,
  Surya-det — zero-shot vs finetuning (macierz eksperymentów).
- **E2** Preprocessing geometryczny/fotometryczny + ablacja 2×2 (raw × preprocessed).
- **E3** Wnioski architektoniczne: CNN vs transformer (bez NMS), granularność detekcji
  (word vs line) jest wyuczona, nie architekturalna.
- **E4** Diagnostyka treningu: LR finder, curvature, timing (`line_benchmark/training_diagnostics/`).
- **E5** Pułapki treningowe: mosaic 2×imgsz (rtdetr), imgsz = najdłuższy bok (detectron2
  MAX_SIZE 1333), dopasowanie batch/imgsz/epochów per model.
- **E6** HPO: sterownik Optuna, per-template subsamples, r2/r3, lr sweep, `lrf`.
- **E7** Ablacja round 2: robustness, leave-one-family-out, kolejki per-rodzina.
- **E8** Wyniki porównawcze (tabela rodzin modeli) — oś porównania.

### F. Rozpoznawanie tekstu (OCR/HTR)
Materiał: `training/ocr/`, `benchmark/`, [[Modele]], [[Fine-tuning_OCR]].
- **F1** Zero-shot test ~12 gotowych modeli (Surya, PARSeq, TrOCR, Tesseract, EasyOCR,
  PaddleOCR, Kraken, RysOCR, GLM-4V, Qwen2.5-VL, GOT-OCR, Calamari) — tabela priorytetów.
- **F2** Fine-tuning Suryi (LoRA): pipeline, ablacja datasetów (v2 vs noA/noB/noC), sweep HP
  (lr×rank), dodanie `--seed` (`training/ocr/surya/`).
- **F3** Fine-tuning PARSeq (docTR): input size 32×128 vs 32×512, interpolacja pozycji ViT,
  retrening (`training/ocr/parseq/`).
- **F4** Fine-tuning TrOCR (transformers 5.0.0): collator, eval bez `predict_with_generate`,
  `save_pretrained` (`training/ocr/trocr/`).
- **F5** Pułapki bibliotek (materiał do rozdziału implementacyjnego): docTR `from_pretrained`
  gubi wagi, rozjazd padding train/serve, `split_wide_crops`, `K ≤ max_label_length-2`,
  transformers 5.0 collator/save, Surya 4D causal mask.
- **F6** Syntetyczna walidacja vs ręczne etykiety: `eval_loss` **odwrotnie** skorelowany z CER —
  kryterium wyboru checkpointu zawodzi.
- **F7** Wyniki porównawcze: tabela EMA/CER/WER (surya, surya_lora, trocr_polish(_ft), parseq_ft).

### G. Generowanie danych syntetycznych
- **G1** Generator danych (fonty imitujące pismo, deformacje, szum) — `benchmark/src/data_generator.py`.
- **G2** Wpływ rodzaju danych syntetycznych na skuteczność (ablacja datasetów Suryi).
- **G3** Kierunki przyszłe: GAN / stable diffusion, ryzyko mode-collapse ([[Zadania]]).

### H. Pipeline end-to-end i aplikacja
- **H1** Aplikacja: upload → prep → detekcja linii → per-line OCR (gradio, `app/`).
- **H2** Warstwa tekstu do zaznaczania nad stroną (SVG overlay).
- **H3** Wybór modelu OCR w aplikacji (model picker z compose).
- **H4** E2E benchmark: cały pipeline vs ground-truth na poziomie strony + tabele Overleaf
  (`e2e_benchmark/`).

### I. Infrastruktura, inżynieria i reprodukowalność
- **I1** Docker + FastAPI per model: GPU, bind mount, HUID/HGID, `HOME=/tmp`, `shm_size`.
- **I2** Orkiestracja treningu: macierz eksperymentów, per-model batch/imgsz/epochy.
- **I3** W&B: dataset fingerprint + commit, group/tags, pułapki uploadu artefaktów.
- **I4** DVC + DagsHub — wersjonowanie i pobieranie danych.
- **I5** `GIT_COMMIT` do kontenerów — reprodukowalność.
- **I6** Ograniczenia dyskowe/partycje (root 99%, cache na `/home`) — decyzje inżynierskie.

### J. Postprocessing wyjścia OCR
Materiał: [[Postprocessing_eksperymenty]] (numery E poniżej), [[Postprocessing_literatura]], `postprocessing/`.
- **J1** Przegląd metod post-OCR: model kanału zaszumionego, słownik, dekodowanie z LM, seq2seq, LLM;
  PolEval 2021 Task 3 (druk).
- **J2** Metodyka: LOGO i zamrożony test, harm / changed-wrong obok CER, istotność, szum strojenia, przeciek.
- **J3** Pułapy zysku (oracle, E1) i pokrycie słownika (E2).
- **J4** Model pomyłek Brill & Moore i jego ablacje (E3, E5): skąd pochodzi zysk.
- **J5** Kompromis CER a bezpieczeństwo: limit szkód (E4), lista terminów i płynne błędne podmiany (E6).
- **J6** Ocena kandydatów przez model OCR patrzący na obraz (E10, wariant docelowy; statyczny jako porównanie).
- **J7** Korektor a jakość modelu OCR (E8) i porównanie z RysOCR (E9).
- **J8** Granice metody: karty pacjenta, recepty, błędy real-word (E12, §8).
- **J9** Wyniki negatywne: pozycja w słowie, wielkość liter, listy medyczne, NER (§7).
- **J10** Implementacja i wdrożenie: skompilowany słownik (23 s → 1.5 s), pakiet `ocr-postprocess`,
  powtarzalny przebieg `rebuild.py`.

