# Pomyłki wieloznakowe (1:2, 2:1, 2:2) — Surya LoRA r64 / v2_200k

- Adapter: `training/results/ocr/surya/v2_200k-lora-r64-mega/adapter`
- Zbiór: handlabeled, **2160** linii, **42305** znaków referencyjnych, 8 źródeł (zeszytów)
- Wyrównanie: Viterbi z kosztami -log P, hard-EM, 5 iteracji; α=0.1, λ=0.5, inwentarz N≥3
- Raportowane: N ≥ 5 i lift ≥ 3.0

> Zastrzeżenia: handlabeled = zbiór selekcji checkpointu (in-sample). „Źródło” to zeszyt/dokument z nazwy pliku — przybliżenie autora, nie jego identyfikator. Kategoria „optyczne?” to heurystyka (wszystko spoza listy reguł ortograficznych) — weryfikować na wycinkach w `examples/`.

## Bilans edycji

| wyrównanie | 1:1 sub | usunięcia | wstawienia | wieloznakowe | edycje (jednostki) |
|---|---:|---:|---:|---:|---:|
| Levenshtein (analyze.py) | 4014 | 914 | 474 | 0 | 5402 |
| ważone + wieloznakowe | 3238 | 518 | 445 | 609 | 4810 |

Zdarzenia wieloznakowe łącznie: **609** (w tym raportowane: 532; wg typu: 1:2 = 64, 2:1 = 412, 2:2 = 56).

## Zdarzenia wieloznakowe (raportowane)

P = N / wystąpienia źródła w GT. lift = P / P(najlepszej dekompozycji na niezależne 1:1/indel). „top źródło” / „top słowo” = udział zeszytu / słowa z największą liczbą zdarzeń (koncentracja: pomyłka ogólna vs jednego autora/słowa).

| # | ref → pred | typ | n | wyst. źródła | P | lift | kategoria | n źródeł | top źródło | n słów | top słowo |
|---:|---|---|---:|---:|---:|---:|---|---:|---|---:|---|
| 1 | `ci` → `u` | 2:1 | 94 | 307 | 30.6% | 934 | optyczne? | 6 | Zrzut ekranu 2026-05-07 (44%) | 30 | kości (32%) |
| 2 | `cz` → `w` | 2:1 | 87 | 517 | 16.8% | 67 | optyczne? | 7 | Zrzut ekranu 2026-05-12 (51%) | 36 | przyczep (16%) |
| 3 | `rz` → `n` | 2:1 | 65 | 673 | 9.7% | 67 | optyczne? | 8 | Zrzut ekranu 2026-05-07 (32%) | 45 | powierzchnia (14%) |
| 4 | `ci` → `w` | 2:1 | 47 | 307 | 15.3% | 273 | optyczne? | 7 | Zrzut ekranu 2026-05-12 (30%) | 28 | kości (21%) |
| 5 | `oc` → `a` | 2:1 | 26 | 169 | 15.4% | 323 | optyczne? | 4 | Zrzut ekranu 2026-05-12 (42%) | 8 | boczny (27%) |
| 6 | `ls` → `b` | 2:1 | 23 | 42 | 54.8% | 3259 | optyczne? | 5 | Zrzut ekranu 2026-05-07 (43%) | 10 | kulszowy (35%) |
| 7 | `k` → `lu` | 1:2 | 22 | 1526 | 1.4% | 4311 | optyczne? | 5 | Zrzut ekranu 2026-05-07 (27%) | 19 | kresa (9%) |
| 8 | `d` → `ol` | 1:2 | 16 | 1302 | 1.2% | 2361 | optyczne? | 5 | anatomia_zmysly (44%) | 16 | dominuje (6%) |
| 9 | `ki` → `lu` | 2:2 | 12 | 200 | 6.0% | 345 | optyczne? | 5 | Noga (42%) | 7 | rzepki (33%) |
| 10 | `rz` → `a` | 2:1 | 11 | 673 | 1.6% | 107 | optyczne? | 3 | Zrzut ekranu 2026-05-07 (45%) | 9 | przyczep (27%) |
| 11 | `śc` → `su` | 2:2 | 11 | 148 | 7.4% | 7 | optyczne? | 4 | Serce - Więcej Niż Lek (55%) | 5 | kości (45%) |
| 12 | `cz` → `wi` | 2:2 | 10 | 517 | 1.9% | 20 | optyczne? | 2 | Zrzut ekranu 2026-05-12 (70%) | 5 | przyczep (60%) |
| 13 | `co` → `w` | 2:1 | 10 | 68 | 14.7% | 684 | optyczne? | 4 | Serce - Więcej Niż Lek (50%) | 10 | nieco (10%) |
| 14 | `rn` → `m` | 2:1 | 9 | 67 | 13.4% | 927 | optyczne? | 5 | data (44%) | 8 | podskórna (22%) |
| 15 | `b` → `lo` | 1:2 | 8 | 544 | 1.5% | 2830 | optyczne? | 5 | anatomia_zmysly (25%) | 7 | brzeg (25%) |
| 16 | `k` → `li` | 1:2 | 7 | 1526 | 0.5% | 335 | optyczne? | 4 | Zrzut ekranu 2026-05-07 (29%) | 6 | międzykłykciowy (29%) |
| 17 | `as` → `o` | 2:1 | 7 | 94 | 7.4% | 25 | optyczne? | 3 | Zrzut ekranu 2026-05-12 (43%) | 6 | zasłonowy (29%) |
| 18 | `ci` → `uj` | 2:2 | 7 | 307 | 2.3% | 246 | optyczne? | 3 | Zrzut ekranu 2026-05-07 (43%) | 6 | nadkłykciowa (29%) |
| 19 | `ks` → `b` | 2:1 | 7 | 48 | 14.6% | 131 | optyczne? | 4 | Noga (43%) | 6 | większy (29%) |
| 20 | `sz` → `r` | 2:1 | 6 | 254 | 2.4% | 46 | optyczne? | 4 | Zrzut ekranu 2026-05-07 (50%) | 5 | większe (33%) |
| 21 | `cz` → `ur` | 2:2 | 6 | 517 | 1.2% | 22 | optyczne? | 3 | Zrzut ekranu 2026-05-12 (50%) | 5 | piszczelowej (33%) |
| 22 | `ą` → `a,` | 1:2 | 6 | 232 | 2.6% | 440 | optyczne? | 2 | Zrzut ekranu 2026-05-12 (50%) | 6 | należą (17%) |
| 23 | `ru` → `ni` | 2:2 | 5 | 85 | 5.9% | 331 | optyczne? | 3 | Noga (60%) | 5 | gruczoły (20%) |
| 24 | `ci` → `a` | 2:1 | 5 | 307 | 1.6% | 444 | optyczne? | 3 | anatomia_zmysly (60%) | 3 | wcięcie (40%) |
| 25 | `tr` → `ł` | 2:1 | 5 | 149 | 3.4% | 14 | optyczne? | 3 | Zrzut ekranu 2026-05-07 (40%) | 4 | strzałkowa (40%) |
| 26 | `łk` → `th` | 2:2 | 5 | 52 | 9.6% | 6 | optyczne? | 2 | Noga (80%) | 4 | strzałkowy (40%) |
| 27 | `ko` → `b` | 2:1 | 5 | 648 | 0.8% | 53 | optyczne? | 4 | Noga (40%) | 5 | skokowa (20%) |
| 28 | `ń` → `ri` | 1:2 | 5 | 102 | 4.9% | 1034 | optyczne? | 1 | Zrzut ekranu 2026-05-12 (100%) | 1 | końcowy (100%) |
| 29 | `rw` → `m` | 2:1 | 5 | 57 | 8.8% | 1302 | optyczne? | 1 | Noga (100%) | 3 | nerw (60%) |

## Substytucje 1:1 — przed vs po oczyszczeniu (top-20 wg Levenshteina)

| ref → pred | Levenshtein | po oczyszczeniu | Δ |
|---|---:|---:|---:|
| `k` → `h` | 448 | 439 | -2% |
| `a` → `o` | 244 | 244 | +0% |
| `i` → `u` | 123 | 20 | -84% |
| `ś` → `s` | 108 | 143 | +32% |
| `t` → `ł` | 102 | 93 | -9% |
| `z` → `w` | 90 | 7 | -92% |
| `c` → `w` | 85 | 72 | -15% |
| `z` → `n` | 67 | 4 | -94% |
| `c` → `u` | 67 | 42 | -37% |
| `o` → `a` | 62 | 59 | -5% |
| `r` → `n` | 58 | 55 | -5% |
| `i` → `w` | 57 | 5 | -91% |
| `k` → `b` | 55 | 56 | +2% |
| `ł` → `t` | 54 | 44 | -19% |
| `z` → `i` | 50 | 39 | -22% |
| `k` → `l` | 49 | 32 | -35% |
| `c` → `s` | 44 | 0 | -100% |
| `z` → `r` | 43 | 36 | -16% |
| `g` → `o` | 40 | 40 | +0% |
| `w` → `u` | 40 | 40 | +0% |

## Top-20 substytucji 1:1 po oczyszczeniu

| ref → pred | liczba | P(pred | ref) |
|---|---:|---:|
| `k` → `h` | 439 | 28.8% |
| `a` → `o` | 244 | 10.0% |
| `ś` → `s` | 143 | 37.0% |
| `t` → `ł` | 93 | 8.4% |
| `c` → `w` | 72 | 5.4% |
| `o` → `a` | 59 | 1.7% |
| `k` → `b` | 56 | 3.7% |
| `r` → `n` | 55 | 3.1% |
| `ł` → `t` | 44 | 5.8% |
| `c` → `u` | 42 | 3.2% |
| `g` → `o` | 40 | 5.6% |
| `w` → `u` | 40 | 1.8% |
| `ż` → `z` | 39 | 28.7% |
| `z` → `i` | 39 | 1.8% |
| `z` → `r` | 36 | 1.7% |
| `ą` → `a` | 33 | 14.2% |
| `s` → `o` | 32 | 2.7% |
| `k` → `l` | 32 | 2.1% |
| `ę` → `e` | 27 | 4.2% |
| `ą` → `o` | 25 | 10.8% |

## Znaki o największym udziale w błędach (po oczyszczeniu)

Błędy znaku = 1:1 + usunięcie + udział w zdarzeniu wieloznakowym (każdy znak źródła).

| znak | wyst. w GT | błędy 1:1/del | w wieloznakowych | % błędów znaku |
|---|---:|---:|---:|---:|
| `k` | 1526 | 620 | 75 | 45.5% |
| `c` | 1325 | 218 | 321 | 40.7% |
| `z` | 2120 | 312 | 202 | 24.2% |
| `a` | 2446 | 311 | 23 | 13.7% |
| `r` | 1765 | 210 | 106 | 17.9% |
| `i` | 2405 | 107 | 172 | 11.6% |
| `s` | 1204 | 132 | 46 | 14.8% |
| `ś` | 387 | 163 | 14 | 45.7% |
| `o` | 3534 | 121 | 51 | 4.9% |
| `t` | 1106 | 153 | 16 | 15.3% |
| `w` | 2213 | 124 | 8 | 6.0% |
| `ł` | 753 | 119 | 12 | 17.4% |
| `·` | 3864 | 105 | 0 | 2.7% |
| `g` | 711 | 85 | 4 | 12.5% |
| `l` | 736 | 60 | 27 | 11.8% |
| `ą` | 232 | 70 | 10 | 34.5% |
| `d` | 1302 | 45 | 20 | 5.0% |
| `ż` | 136 | 61 | 0 | 44.9% |
| `ę` | 650 | 55 | 3 | 8.9% |
| `e` | 2306 | 53 | 0 | 2.3% |

## Zbieżność EM

| iteracja | koszt całkowity | zdarzenia wieloznakowe | inwentarz |
|---:|---:|---:|---:|
| 0 | 5402.0 | 0 | 0 |
| 1 | 23707.4 | 606 | 54 |
| 2 | 23837.6 | 609 | 50 |
| 3 | 23810.4 | 609 | 50 |
| 4 | 23807.6 | 609 | 50 |
| 5 | 23807.6 | 609 | 50 |

## Pliki

- `multi_events.csv`, `sub_before_after.csv`, `unit_error_stats.csv`, `confusion_counts_clean.csv`
- `heatmap_clean.png`, `heatmap_clean_top.png`
- `examples/INDEX.md`
