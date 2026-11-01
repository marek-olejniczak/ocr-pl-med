# Generator linii OCR v3: kształty liter

Analiza pomyłek najlepszego modelu v2 (Marek, gałąź `error-analysis`,
`notes/vault/Generator_v3_wskazowki.md`) i jej sprawdzenie na 17 modelach
(`notes/vault/Generator_v3_odpowiedz.md`) dały jeden wniosek: zostały błędy
**kształtu**, nie ekspozycji. v2 miało 2–6× więcej „ci”, „cz”, „rz”, „ś” niż
v1, a pomyłki `ci→u`, `cz→w` się nie ruszyły. Fine-tuning na syntetykach wręcz
dołożył część z nich (`cz→w` 25 → 86, `t→ł` 4 → 153 względem modelu bazowego).
Dlatego v3 nie dokłada treści ani upsamplingu, tylko trzy rodziny zmieniające
wygląd liter. Kod: `src/line_glyphs.py`.

## Trzy nowe rodziny

| rodzina | co robi | w ilu liniach | na co celuje |
|---|---|---:|---|
| `cursive_joins` | litery się stykają: tracking −0.06…0.03 rozmiaru fontu zamiast +0.06…0.20, szerokość glifu 0.95–1.10 zamiast 1.05–1.25; pary `ci, cz, rz, oc, ls, rn, co, sz, ks, ko, as, ru, rw, tr, ki, śc` dosunięte o dodatkowe 2–9% rozmiaru (z p = 0.7) | 40% | `ci→u`, `cz→w`, `rz→n`, `ls→b`, `rn→m` |
| `glyph_variants` | wybrane litery linii (`k, a, t, ł, g, r, o`, każda z p = 0.4) rysowane innym fontem: takim, w którym ta litera najbardziej przypomina tę, z którą model ją myli. Cała linia pisze daną literę tak samo | 50% | `k→h`, `a→o`, `t↔ł`, `g→o`, `r→n`, `o→a` |
| `diacritic_placement` | kreska, kropka lub ogonek (`ąćęńóśźż`, kropka nad `i`/`j`) przesunięta (zwykle w prawo i w górę, obrót ±30°, skala 0.7–1.4), ogonek oderwany w dół-prawo, albo znak pominięty (12%). Etykieta zostaje poprawna ortograficznie | 50% linii, 60% znaków w nich | `ś→s`, `ż→z`, `ą→a`, `ą→a,`, część `ci→u` |

Każdą wyłącza `--disable <nazwa>`, jak rodziny v2.

### Jak dobierane są „mylące” litery

Dla każdej pary (np. `k`/`h`) i każdego z 35 fontów: obie litery rysowane
w tym samym foncie, szkielet (ślad pióra bez grubości) w ramce 32×32,
podobieństwo = IoU szkieletów. Do puli trafia 8 fontów o najwyższym wyniku.
Szkielet zamiast pełnej maski, bo pełna maska wybierała same grube fonty
(gruba kreska pokrywa się z czymkolwiek). Dawca musi mieć podobną grubość
pióra co font linii (stosunek ≤ 1.3), a jego rozmiar dobiera średnia
geometryczna wysokości „x” i wysokości samej litery.

Pule dla `k` (podobne do `h`): MarckScript, DancingScript, Charm, PlaywriteCZ,
PlaywriteAT, Courgette, PlaywriteHU, PlaywriteHR, czyli pętlowe kursywne „k”.

### Jak wycinany jest znak diakrytyczny

Litera i jej litera bazowa (`ś` i `s`) rysowane w tym samym foncie od tego
samego punktu bazowego; znak = piksele `ś` poza (pogrubionym o 2 px) `s`.
Kropka nad `i`/`j` = najwyżej położona osobna składowa. Przesunięty znak nie
rusza litery, bo wycinek jest centrowany na pierwotnym obrysie litery.

## Parowanie zbiorów (`--line-seeds`)

Każda nowa rodzina losuje z własnego generatora, ustawianego od tekstu linii,
fontu i rozmiaru. To nie wystarcza: węższa linia (zlepienie) to mniej linii
kratki do narysowania, więc mniej losowań i reszta zbioru się rozjeżdża.
Dlatego `--line-seeds`: przed każdą linią, i przed każdym jej etapem (papier,
sąsiednie linie, morfologia, elastic, aparat), generator jest ustawiany od
nowa z `(seed, numer linii)`. Skutek:

- pięć zbiorów pilota ma identyczny tekst, font, rozmiar, podział train/val
  i wszystkie pozostałe efekty linia w linię; różnią się tylko rodzinami
  glifów (sprawdzone na 200 liniach: 200/200),
- zbiór można generować w kawałkach równolegle (`--start-index`,
  `build_ablation_sets.py generate --jobs N`), wynik jest bit w bit taki sam
  jak z jednego przebiegu (sprawdzone na 59 liniach w 3 kawałkach),
- bez `--line-seeds` generator odtwarza stare zbiory jak dotąd
  (`v2_200k`: tekst i metryka 300/300).

## Pilot

| zbiór | rodziny glifów | po co |
|---|---|---|
| `v3_base_200k` | żadna | odniesienie (to samo co v2, inny seed: 2031) |
| `v3_200k` | wszystkie trzy | czy v3 pomaga |
| `v3_no_cursive_joins_200k` | bez łączeń | ile daje każda rodzina |
| `v3_no_glyph_variants_200k` | bez wariantów | |
| `v3_no_diacritic_placement_200k` | bez diakrytyków | |

Treningi jak w ablacji (r64, batch 8, 10 000 kroków, ~25 min na RTX 3090),
seed 42. Do tego `v3_base_200k` i `v3_200k` drugi raz z seedem 43, żeby znać
szum między dwoma treningami na tych samych danych. Razem 7 treningów.

Ocena na **roboczej** części `handlabeled` (1212 linii, `benchmark/splits/`):
CER i liczba konkretnych pomyłek (`ci→u`, `k→h`, `ś→s` …) oraz kierunków
odwrotnych (`h→k`, `u→ci`). Zamrożona część (948 linii) dopiero na koniec, dla
wybranej konfiguracji.

Instrukcja uruchomienia: `docs/v3_runbook.md`.
