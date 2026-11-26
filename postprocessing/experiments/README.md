# Eksperymenty i przeliczanie od zera

Wszystkie polecenia uruchamia się z katalogu `postprocessing/experiments`.

```bash
cd postprocessing/experiments
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt   # instaluje też ../package (-e)
source ../.venv/bin/activate
ocr-postprocess setup --all          # słowniki, w tym sjp (potrzebne ablacjom), i skompilowany indeks
pytest -q ../package/tests && pytest -q tests
```

## Przeliczenie wszystkiego na nowym (finalnym) zbiorze testowym

### Co trzeba mieć

| co | format |
|---|---|
| zbiór | katalog z `manifest.csv` (`page_id, document, split, original_name, …`; `split` = `dev` / `test`) i `crops.csv` (`file_name, label, line_id`) |
| przebieg benchmarku | `<run>/<model>/<dataset_id>/raw_predictions.csv` (`file_name, ground_truth, prediction`) dla modelu głównego i porównywanych |
| adapter Suryi | ten sam, którym powstały predykcje modelu głównego |
| obrazy linii na maszynie z GPU | płaski katalog `<images_prefix><file_name>` |

Etykiety w przebiegu muszą się zgadzać z `crops.csv`, inaczej import przerywa. Linie `[BLANK]` są pomijane.

### Kroki

1. **Konfiguracja:** skopiuj `configs/v12.json` do `configs/final.json`, zmień `name`, ścieżki, modele i
   `bundle_name`. Nieznane pole to błąd (literówka nie przejdzie po cichu).
2. **Grupy:** jeśli zbiór ma nowe źródła, dopisz je w `data/group_rules.json`. **Grupa = jedno źródło / autor**:
   to jednostka, której korektor nie widzi przy swojej ocenie. Cała grupa musi być w `dev` albo w `test`.
3. **Ustal przed uruchomieniem** (i nie zmieniaj po obejrzeniu testu): `harm_max`, `domain`, `compare_models`.
4. Lokalnie:
   ```bash
   python rebuild.py configs/final.json prepare
   ```
   Przejrzyj wypisaną tabelę grup, podział dev/test i surowy CER. Dopiero potem:
   ```bash
   python rebuild.py configs/final.json static candidates
   ```
5. **Na GPU:** ocena kandydatów przez Suryę. Etap `candidates` wypisuje gotowe polecenie
   (`evaluation/surya_lp.py`). Plik `req.jsonl` zawiera tekst linii. Wynik `resC.jsonl` skopiuj tam, gdzie wskazuje
   polecenie (`$PP_CACHE/runs/<name>/lp/`).
6. Lokalnie:
   ```bash
   python rebuild.py configs/final.json rescore bundle ablations report
   ```
7. Wynik: `../results/<name>/SUMMARY.md`. Jeśli nowa paczka ma trafić do aplikacji, ustaw `DEFAULT_BUNDLE`
   w `../package/src/ocr_postprocess/bundle.py`, uruchom testy i `ocr-postprocess check`, zaktualizuj tabelę
   w `../package/README.md` (§3) i linię kontrolną w `__main__.py`, jeśli przestała przechodzić.
8. Do repo: `data/<name>/groups.csv`, `../results/<name>/`, nowa paczka i konfiguracja. Nic z `$PP_CACHE`.

### Etapy

| etap | co liczy | czas (12 wątków) |
|---|---|---|
| `prepare` | import predykcji, mapa grup i split, surowy CER/WER wszystkich modeli | sekundy |
| `static` | dla modelu głównego i porównywanych: LOGO i dev → test (pełna lista kandydatów); progi `deploy_params.json` | ok. 30 min na model |
| `candidates` | kandydaci do oceny przez Suryę (`req.jsonl`) | ok. 2 min |
| *GPU* | log P wariantów linii (v1.2: ok. 2900 linii; RTX 3090 ok. 100 tekstów/s) | minuty |
| `rescore` | top-5 bez Suryi (`rescore_w0.json`) i z Suryą (`rescore_surya.json`), LOGO i test; progi do paczki | ok. 45 min |
| `bundle` | paczka w `ocr_postprocess/models/<bundle_name>/` | ok. 1 min |
| `ablations` | oracle, pokrycie słownika, model pomyłek vs Levenshtein, 13 wariantów LOGO (`results/<name>/ablations/`) | ok. 4 h |
| `report` | `SUMMARY.md` | sekundy |

Czasy zmierzone na v1.2 (2904 linie, 11 grup) na 12 wątkach laptopa. Rosną z liczbą grup (strojenie jest
zagnieżdżone). Uśpienie komputera wstrzymuje obliczenia, ale ich nie psuje.

**Sprawdzone 2026-11-26:** `prepare` → `bundle` od surowego przebiegu benchmarku (ocena Suryi z zapisanego
`resC.jsonl`) odtworzyło `results/v12`: surowy CER, wyniki z oceną Suryi i bez, protokół dev → test i progi
identycznie, model pomyłek w paczce bajt w bajt. Jedyna różnica to pełna lista kandydatów w LOGO: −1.22 zamiast
−1.23 p.p., bo zapisany plik liczono na węższej siatce progów.

`--dry-run` wypisuje polecenia bez uruchamiania. Etap nie nadpisuje istniejących wyników bez `--force`.
Logi: `$PP_CACHE/runs/<name>/logs/`.

### Co sprawdzić w wynikach

- **Brzeg siatki:** w logach `rescore_surya` i `static_*` zobacz wybrane progi. `delta` = −16 albo `w` = 2 to
  brzeg (tak było na v1.2). Siatki są w `evaluation/run_cv.py` (`GRID`) i `run_rescore.py` (`GRID`, `--ws`).
  Poszerzenie po obejrzeniu testu trzeba opisać w pracy.
- **Test liczy się przy każdym** uruchomieniu `static` i `rescore`. Do porównywania wariantów używaj LOGO na dev
  (`run_holdout` bez `--test`).
- `rebuild.py` **nie** obejmuje pomiarów jednorazowych: RysOCR (wrapper na serwerze; jego predykcje można dodać do
  `compare_models`, jeśli są w przebiegu benchmarku), zbiorów RxHandBD i htrlex, analizy kart pacjenta,
  dekodowania Suryi (lookahead) i wyników negatywnych. Które to: notatka w vault, §6 i §7.

## Pojedyncze narzędzia (`evaluation/`)

Każde ma `--help`.

| moduł | co |
|---|---|
| `run_cv`, `run_holdout`, `run_rescore` | trzy protokoły: LOGO, dev → test, ocena z Suryą |
| `rescore_cands`, `surya_lp` | kandydaci → log P z Suryi (GPU) |
| `build_bundle` | paczka z predykcji i progów |
| `import_bench`, `raw_ocr`, `report` | wejście z benchmarku, surowy CER, `SUMMARY.md` |
| `oracle`, `p2_report`, `p3_report` | pułapy, słownik i klasy chronione, model pomyłek |
| `metrics`, `compare`, `sig`, `folds` | CER/WER, tabela tokenów, istotność, grupy |
| `build_domain` | lista terminów anatomicznych z Wikidata (`data/wikidata_ta.rq`) |
| `make_table` | tylko archiwalna `results/dataset-v1/TABLE.md` |

Format predykcji: jsonl z polami `file_name`, `ground_truth`, `prediction`. Raporty `oracle`, `p2_report`,
`p3_report` bez `--predictions` liczą na dataset-v1 i potrzebują `$PP_CACHE/dataset-v1/predictions.jsonl`
(tekst notatek, poza repo; bez niego odpowiednie testy są pomijane).
