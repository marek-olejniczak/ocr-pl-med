# Postprocessing wyjścia OCR

Korekta linii z Suryi słownikiem i modelem pomyłek, z opcjonalną oceną kandydatów przez samą Suryę.
Na test/real/v1.2 (Surya `v4_200k_25k`): CER 0.1098 → 0.0858 z oceną Suryi, → 0.0939 bez niej.
Liczby zostaną przeliczone na finalnym zbiorze testowym.

| chcę… | idę do |
|---|---|
| **podłączyć korektor do aplikacji** | [`package/README.md`](package/README.md) i przykład [`package/examples/surya_worker.py`](package/examples/surya_worker.py) |
| **przeliczyć wszystko na nowym zbiorze testowym** | [`experiments/README.md`](experiments/README.md) |
| znaleźć liczby i raporty | [`results/README.md`](results/README.md) |
| opisać to w pracy | [`../notes/vault/Postprocessing_eksperymenty.md`](../notes/vault/Postprocessing_eksperymenty.md) |

```
package/       instalowalny pakiet `ocr-postprocess` (tego potrzebuje aplikacja; nie zależy od reszty)
experiments/   ewaluacja, strojenie, budowa paczki; `rebuild.py` liczy wszystko od zera
results/       same liczby: v12/ (aktualne), dataset-v1/ (archiwum)
```

**Zasada o danych:** tekst linii (etykiety, predykcje, kandydaci, wycinki) nie trafia do repo. Leży w `$PP_CACHE`
(domyślnie `~/.cache/ocr-pl-med/postprocessing`). W repo są liczby, mapy grup po nazwach plików i pojedyncze słowa
w tabelach pomyłek i przykładach.

## Powiązane części repo

- `training/ocr/error_analysis/`: analiza pomyłek znaków Suryi; punkt wyjścia modelu pomyłek.
- `training/ocr/surya/`: trening adaptera; `cer_callback.py` wybiera checkpoint po CER na etykietach ręcznych.

Starsze commity i raporty używają dawnego układu `training/ocr/postprocessing/` (moduł `pp`, `deploy/`,
`pp_worker.py`, `fetch_resources.py`). Odpowiedniki: `package/src/ocr_postprocess/` (`pp.deploy` → `bundle`,
`pp_worker.py` → `server.py`, `fetch_resources.py` → `resources.py`), `deploy/` → `ocr_postprocess/models/`.
„Tickety” w tekstach to lokalne notatki robocze spoza repo.
