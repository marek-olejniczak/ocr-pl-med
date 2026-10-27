# Pomyłki znaków — Surya LoRA r64 / v2_200k (best checkpoint)

- Adapter: `training/results/ocr/surya/v2_200k-lora-r64-mega/adapter`
- Zbiór: handlabeled, **2160** linii, **42305** znaków referencyjnych
- CER = `0.127692`  WER = `0.481740`  EMA = `0.2528`

> Zastrzeżenie: handlabeled to **ten sam** zbiór, po którym wybierano checkpoint
> (callback `eval_cer`), więc analiza jest „in-sample” względem selekcji. Struktura
> pomyłek jest mimo to diagnostyczna. Rzadkie znaki (Ą 13, Ć 4, Ź 5 wystąpień)
> mają zbyt mało prób, by ich statystyki były stabilne. Spacja pokazywana jako `·`.

## Bilans edycji (całe 2160 linii)

- substytucje: **4014**
- usunięcia: **914**
- wstawienia: **474**
- suma edycji: **5402**

## Top-20 pomyłek (substytucje)

| # | ref → pred | liczba | P(pred \| błąd ref) |
|---|---|---:|---:|
| 1 | `k` → `h` | 448 | 67.0% |
| 2 | `a` → `o` | 244 | 77.0% |
| 3 | `i` → `u` | 123 | 47.3% |
| 4 | `ś` → `s` | 108 | 81.8% |
| 5 | `t` → `ł` | 102 | 64.2% |
| 6 | `z` → `w` | 90 | 21.8% |
| 7 | `c` → `w` | 85 | 26.2% |
| 8 | `z` → `n` | 67 | 16.3% |
| 9 | `c` → `u` | 67 | 20.7% |
| 10 | `o` → `a` | 62 | 45.6% |
| 11 | `r` → `n` | 58 | 31.7% |
| 12 | `i` → `w` | 57 | 21.9% |
| 13 | `k` → `b` | 55 | 8.2% |
| 14 | `ł` → `t` | 54 | 42.9% |
| 15 | `z` → `i` | 50 | 12.1% |
| 16 | `k` → `l` | 49 | 7.3% |
| 17 | `c` → `s` | 44 | 13.6% |
| 18 | `z` → `r` | 43 | 10.4% |
| 19 | `w` → `u` | 40 | 30.8% |
| 20 | `g` → `o` | 40 | 48.8% |

## Najgorsze znaki (min. 10 wystąpień, wg odsetka błędów)

| znak | wystąpień | odsetek błędów (sub+del) | top pomyłka |
|---|---:|---:|---|
| `/` | 18 | 50.0% | `1` (7) |
| `ś` | 387 | 45.7% | `s` (108) |
| `k` | 1526 | 45.6% | `h` (448) |
| `ż` | 136 | 44.9% | `z` (36) |
| `c` | 1325 | 40.7% | `w` (85) |
| `ą` | 232 | 34.5% | `a` (31) |
| `ź` | 74 | 29.7% | `ć` (6) |
| `J` | 35 | 25.7% | `g` (6) |
| `,` | 82 | 25.6% | `.` (2) |
| `z` | 2120 | 24.3% | `w` (90) |
| `:` | 66 | 21.2% | `,` (2) |
| `(` | 50 | 18.0% | `C` (4) |
| `r` | 1765 | 18.0% | `n` (58) |
| `K` | 163 | 17.8% | `W` (9) |
| `ł` | 753 | 17.4% | `t` (54) |

## Diakrytyki

| para | baza→diakrytyk | diakrytyk→baza |
|---|---:|---:|
| `a`↔`ą` | 5 (2%) | 31 (39%) |
| `c`↔`ć` | 8 (2%) | 2 (20%) |
| `e`↔`ę` | 11 (25%) | 23 (42%) |
| `l`↔`ł` | 18 (33%) | 5 (4%) |
| `n`↔`ń` | 1 (2%) | 1 (7%) |
| `o`↔`ó` | 10 (7%) | 20 (56%) |
| `s`↔`ś` | 15 (11%) | 108 (82%) |
| `z`↔`ź` | 1 (0%) | 4 (19%) |
| `z`↔`ż` | 6 (1%) | 36 (61%) |
| `A`↔`Ą` | 1 (17%) | 1 (100%) |
| `C`↔`Ć` | 0 | 0 |
| `E`↔`Ę` | 4 (40%) | 3 (75%) |
| `L`↔`Ł` | 0 (0%) | 1 (12%) |
| `N`↔`Ń` | 0 (0%) | 1 (100%) |
| `O`↔`Ó` | 1 (12%) | 4 (80%) |
| `S`↔`Ś` | 2 (40%) | 2 (40%) |
| `Z`↔`Ź` | 1 | 0 |
| `Z`↔`Ż` | 1 (8%) | 3 (75%) |

## Pliki

- `heatmap.png`
- `heatmap_top.png`
- `small_multiples.png`
- `diacritics.png`
- `confusions.html`
- `confusion_counts.csv`
- `confusion_error_prob.csv`
- `symbol_stats.csv`
- `top_confusions_per_symbol.csv`
- `del_ins_stats.csv`
- `examples/INDEX.md`

