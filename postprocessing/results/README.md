# Wyniki: który plik co zawiera

Same liczby, bez tekstu linii (wyjątek: pojedyncze słowa w tabelach pomyłek i listach podejrzanych etykiet).
Opis eksperymentów i co cytować: `../../notes/vault/Postprocessing_eksperymenty.md`.

## `v12/`: aktualne (test/real/v1.2, 2904 linie, 11 grup)

Czytać zaczynając od [`v12/README.md`](v12/README.md).

| plik | co |
|---|---|
| `raw_ocr.csv` | CER/WER 7 modeli OCR bez korekty |
| `rescore_surya.json` | **wariant docelowy** (top-5 + ocena Suryi): `logo`, `test`, `params_dev` |
| `rescore_w0.json` | top-5 bez Suryi (statyczny, tryb zapasowy) |
| `p4_summary.csv` | pełna lista kandydatów, LOGO, po wierszu na model (v2, v3, v4, v4_25k, RysOCR) |
| `p4_v12_<model>_ta_fine_folds.csv`, `…_per_group.json` | progi i wynik per grupa; CER per grupa przed i po |
| `holdout_v12_<model>_ta.json` | pełna lista kandydatów, dev → test |
| `deploy_params_top5.json`, `deploy_params_surya.json` | progi w paczce `v4_200k_25k` (statyczne, z oceną) |
| `deploy_params.json` | progi dla pełnej listy kandydatów (nieużywane w paczce) |

## `dataset-v1/`: archiwum (Surya v2_200k, 2160 linii, 8 grup)

Tu są ablacje i analizy metody, których na v1.2 nie liczono. [`RAPORT.md`](dataset-v1/RAPORT.md) to całość,
`TABLE.md` tabela, `ORACLE.md` / `P2_P3.md` / `P4.md` fazy. Pliki `p4_option1_*` i `p4_abl_*` pochodzą ze
starszych predykcji i etykiet (inny baseline niż reszta). `holdout_*.json`: `base` bez listy terminów, `ta` z listą,
`ta_tuned` z listą także w strojeniu. Polecenia i ścieżki w tych raportach są z dawnego układu katalogów.

## Nowe przebiegi

`rebuild.py` tworzy `<name>/` z `SUMMARY.md` na wierzchu, `config.json`, tymi samymi rodzajami plików co wyżej
(tag = nazwa modelu) i podkatalogiem `ablations/`.
