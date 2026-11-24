# Surya v4 i test/real/v1.2 (2026-11-24)

Predykcje: benchmark serwera `ocr-v4_20261123_140539` (7 modeli na real_v1.2), import `evaluation.import_bench`.
v1.2 = v1.1 (etykiety bez zmian) + 80 stron `new` (541 linii): karty pacjenta 1–3 i recepta (141 linii, G9_karta),
„Nerw Twarzowy” (188, G10), „Żyły Głębokie” (212, G11). 68 linii `[BLANK]` zostaje w zbiorze dla detekcji, w ocenie OCR
je pomijamy (`metrics.load_predictions`): 2904 linie. Dev/test z manifest.csv = TEST_GROUPS z `run_holdout`
(noga, anatomia_zmysly, miednica): dev 1835, test 1069.

## Surowy OCR (`raw_ocr.csv`)

| model | całość | dev | test | karty (G9) | nowe notatki (G10+G11) |
|---|---|---|---|---|---|
| surya (bez LoRA) | 0.2251 | 0.2208 | 0.2347 | 0.524 | 0.202 |
| ocr800k | 0.1991 | 0.1942 | 0.2101 | 0.587 | 0.155 |
| v2_200k | 0.1550 | 0.1566 | 0.1513 | 0.608 | 0.116 |
| v3_base_200k | 0.1499 | 0.1519 | 0.1454 | 0.507 | 0.115 |
| v3_200k | 0.1446 | 0.1465 | 0.1405 | 0.526 | 0.112 |
| v4_200k (10k kroków) | 0.1160 | 0.1190 | 0.1091 | 0.592 | 0.080 |
| v4_200k_25k | **0.1098** | 0.1124 | 0.1040 | 0.493 | 0.074 |

## Korektor, wariant główny (harm ≤ 0.3%, lista TA `en,la,pl`)

LOGO po 11 grupach (`p4_v12_*`, `p4_summary.csv`) i protokół dev → test (`holdout_v12_*`).

| model | LOGO | Δ | harm | fixed / broken / changed_wrong | test (progi z dev) | Δ test | harm test |
|---|---|---|---|---|---|---|---|
| v2_200k | 0.1550 → 0.1297 | −2.53 p.p. (−16.3%) | 0.30% | 1176 / 15 / 241 | 0.1513 → 0.1311 | −2.03 | 0.06% |
| v3_200k | 0.1446 → 0.1200 | −2.47 (−17.0%) | 0.23% | 1140 / 12 / 255 | 0.1405 → 0.1204 | −2.01 | 0.00% |
| v4_200k | 0.1160 → 0.1025 | −1.35 (−11.6%) | 0.20% | 751 / 12 / 220 | 0.1091 → 0.0981 | −1.10 | 0.00% |
| v4_200k_25k | 0.1098 → **0.0975** | −1.23 (−11.2%) | 0.27% | 688 / 16 / 185 | 0.1040 → **0.0956** | −0.84 | 0.05% |

CI po grupach dla v4_200k_25k: [−1.45, −0.96] p.p.

Wnioski:
- v4_200k_25k bez korekty (0.1098) jest lepszy niż v3 z korektorem (0.1200). Korektor dalej daje istotny zysk,
  ale mniej więcej o połowę mniejszy niż na v2/v3.
- Zasięg: złych tokenów kwalifikujących się do korekty 2453 (v3: 3292), GT wśród kandydatów 1797 (v3: 2458),
  naprawione 688, czyli 38% osiągalnych (v3: 46%). 41% tokenów naprawianych przez korektor na v3 v4 czyta sam.
- Strojenie wybiera prawie zawsze `L_min = 6` (na v2/v3 3–4): na v4 korekta krótkich słów przestała się opłacać.
  Progi do wdrożenia dla v4: `deploy_params.json` (`lam 0.75, delta −4, mu 3, d_max 3, L_min 6, f_min 0`).
- Profil złych tokenów v4 jak w v3: ok. 81% to inne błędy, 6% tylko diakrytyki, 5% interpunkcja, 1% wielkość liter.
- Karty pacjenta (G9): CER ok. 0.5 we wszystkich modelach, korektor prawie bez efektu (GT wśród kandydatów
  dla 31 z 80 złych tokenów): krótkie pola, nazwiska, daty, kody ICD, nazwy leków. W kilku liniach predykcja zawiera
  nadruk formularza, którego nie ma w etykiecie: możliwy rozjazd wycinka i etykiety.

Odtworzenie (jsonl z tekstem linii poza repo):

```bash
export PYTHONPATH=.
P=~/.cache/ocr-pl-med/postprocessing/preds-v4
.venv/bin/python -m evaluation.import_bench --run <benchmark/wyniki/ocr-v4_...> --dataset-dir <dataset/test/real/v1.2> --out $P
.venv/bin/python -m evaluation.run_cv --harm-max 0.003 --domain en,la,pl --predictions $P/surya_lora_v4_200k_25k.v12.jsonl \
    --groups data/v12/groups.csv --tag v12_v4_200k_25k_ta --out-dir results/v12
.venv/bin/python -m evaluation.run_holdout --harm-max 0.003 --domain en,la,pl --test --predictions $P/surya_lora_v4_200k_25k.v12.jsonl \
    --groups data/v12/groups.csv --tag v12_v4_200k_25k_ta --out-dir results/v12
```
