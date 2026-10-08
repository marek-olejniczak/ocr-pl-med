# Wyniki rundy 2: każda augmentacja osobno

Benchmarki: `ocr-odpornosc_20261007_235959` (zbiory kontrolne) i
`ocr-ablacja-rodziny_20261008_051422` (treningi bez jednej rodziny).
Pełne tabele: `docs/odpornosc.md`, `docs/wyniki_ablacji_rodziny.md`,
`docs/val_breakdown.md`. Metoda: `docs/ablacja_runda2.md`.

Kontrola: `v2_200k` dał w obu rundach identyczne predykcje na 2160 z 2160
linii. Odczyt jest deterministyczny, więc obie rundy da się porównywać wprost.

## 1. Co przeszkadza modelowi czytać

Zbiór kontrolny: 1500 linii, każda w 31 wersjach różniących się jedną
augmentacją. Zmiana CER względem czystej wersji tej samej linii, model v2.

| Augmentacja | Najsłabsza siła | Średnia | Najmocniejsza | v1 przy najmocniejszej |
|---|---|---|---|---|
| sąsiednie linie | +0,1 | +0,7 * | **+8,6 *** | +30,7 * |
| zdjęcie telefonem (wszystko razem) | 0,0 | +0,2 * | **+3,7 *** | +5,3 * |
| dylatacja tuszu | +0,1 | | +1,1 * | +2,3 * |
| erozja tuszu | −0,1 | 0,0 | +1,0 * | +1,6 * |
| grid distortion | 0,0 | +0,1 | +0,2 * | +0,3 * |
| fotokopia (skaner v1) | | | +0,2 * | +0,3 * |
| elastic | +0,1 | +0,1 | +0,1 | +0,3 * |
| kratka | 0,0 | 0,0 | +0,1 | +0,3 * |

Punkty procentowe CER, `*` oznacza 95% przedział bez zera. Czysta wersja:
v1 1,6%, v2 0,6%.

Składowe zdjęcia z telefonu osobno, każda na najmocniejszym poziomie:
pomniejszenie +0,4, rozmycie +0,2, poruszenie +0,2, a cień, szum i JPEG
po 0,0. Osobno prawie nic, razem +3,7. Zniekształcenia się kumulują.

**Wnioski.** Najbardziej problematyczne są sąsiednie linie wchodzące
głęboko w kadr. Tylko przy najmocniejszej sile, ale wtedy wyraźnie: v2 traci
8,6 punktu, v1 aż 30,7. Druga jest mocno zdegradowana fotografia. Elastic,
grid distortion i kratka są dla Suryi praktycznie obojętne nawet przy
maksymalnej sile z naszego zakresu.

## 2. Co pomaga w treningu

Dziewięć treningów, każdy bez jednej rodziny, porównanych linia po linii
z `v2_200k` (CER 12,72%) na 2160 prawdziwych liniach.

| Wyłączona rodzina | CER | Zmiana | 95% przedział |
|---|---|---|---|
| słownictwo anatomiczne | 14,7% | **+1,93 *** | +1,57 .. +2,32 |
| wersaliki | 14,5% | **+1,75 *** | +1,25 .. +2,27 |
| sąsiednie linie | 13,2% | +0,47 * | +0,22 .. +0,76 |
| kratka | 12,9% | +0,16 | −0,00 .. +0,34 |
| morfologia | 12,8% | +0,12 | −0,05 .. +0,28 |
| strzałki i punktory | 12,8% | +0,07 | −0,12 .. +0,26 |
| elastic | 12,8% | +0,06 | −0,10 .. +0,22 |
| zdjęcie telefonem | 12,8% | +0,06 | −0,12 .. +0,24 |
| krótkie słowa | 12,6% | −0,13 | −0,33 .. +0,07 |

**Słownictwo anatomiczne (+1,93).** Wynik nie jest czysty. Bez puli
anatomicznej generator wraca do treści formularzowej, w której nie ma
nagłówków, krótkich słów ani strzałek. Ten wariant to w praktyce cała grupa
treści naraz i zgadza się z nią: grupa A w rundzie 1 dała +2,02.

**Wersaliki (+1,75).** Cały efekt to wielkość liter. Przy porównaniu bez
rozróżniania wielkości różnica znika (−0,06). Na 217 liniach pisanych
wielkimi literami model bez tej rodziny myli się o 26 punktów częściej: czyta
litery poprawnie, ale zapisuje je małymi, na przykład `MIĘŚNIE UDA` jako
`Mięśnie uda`. Czy to ważne, zależy od zastosowania. Dla wiernej transkrypcji
tak, dla wyszukiwania terminów raczej nie.

**Sąsiednie linie (+0,47).** Istotne w tym porównaniu, ale niespójne
z rundą 1, gdzie wyłączenie sąsiednich linii razem z kratką dało tylko +0,10.
Przedziały obejmują wyłącznie losowość zbioru testowego. Każdy wariant
trenowaliśmy raz, więc losowość samego treningu nie jest zmierzona, a ta
niespójność sugeruje, że może sięgać pół punktu. Efekt jest prawdopodobny,
bo sąsiednie linie to też najgroźniejszy warunek z punktu 1, ale wymaga
powtórzenia.

**Reszta, w tym elastic,** nie daje mierzalnego efektu w żadną stronę.

**Krótkie słowa** pomagają tylko najkrótszym liniom. Na liniach do 5 znaków
bez tej rodziny CER rośnie z 25,2% do 27,3%, ale to 199 linii, różnica mieści
się w szumie i na całości nic nie zmienia. Zysk na krótkich liniach z rundy 1
pochodził więc głównie ze słownictwa i wersalików, a nie z tej rodziny.

## 3. Najtrudniejsze kroje pisma

Na 10 000 linii walidacyjnych (`docs/val_breakdown.md`) model v2 najgorzej
radzi sobie z krojami Marek_6 (6,1%), Marek_5 (5,2%), Marek_4 (4,2%) i
Marek_3 (3,8%). To kroje z prawdziwego pisma, najbliższe zbiorowi testowemu.
Kroje z Google Fonts mają po 0,1% do 2%. Model v1 nie widział nowych krojów
w treningu, więc jego wyniki na tym zbiorze nie są porównywalne.

## 4. Co z tego wynika

1. Treść jest najważniejsza: słownictwo i wielkość liter. To potwierdza
   rundę 1 i pokazuje, które elementy treści się liczą.
2. Elastic nie pomaga i nie przeszkadza. Surya jest na te zniekształcenia
   niewrażliwa w całym zakresie, którego używamy. To odpowiedź na pytanie
   promotora.
3. Jedyna augmentacja obrazu z sygnałem to sąsiednie linie. To zarazem
   najgroźniejszy warunek przy odczycie.

## 5. Otwarte

- **Losowość treningu.** Każdy wariant trenowany raz. `cli.py` nie ma
  opcji `--seed`, więc dwa treningi z różnym seedem wymagają drobnej zmiany
  w kodzie Marka. Dwa przebiegi `v2_200k` i dwa `no_neighbour_glyphs_200k`,
  razem około 100 minut, rozstrzygną efekt sąsiednich linii.
- **Czy model bez danej augmentacji jest na nią mniej odporny.** Gotowa
  konfiguracja `experiments_rodziny_odpornosc.yaml`, około 2 godzin bez
  treningu. Pokaże na przykład, czy model bez sąsiednich linii gorzej czyta
  linie z sąsiadami.
