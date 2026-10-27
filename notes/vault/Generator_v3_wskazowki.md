Wskazówki do generatora danych v3 dla OCR linii, oparte na analizie pomyłek najlepszego modelu (Surya LoRA r64 / `v2_200k`, CER 0.1277). Notatka opisuje przede wszystkim **co i jak można generować**. Wady fontów i robienie nowych są na końcu jako wnioski poboczne.

#dane_v3 #generator #analiza #surya

Źródła liczb: `training/ocr/error_analysis/` (README, `results/chars`, `results/ngram`), branch `error-analysis`. Generator v2: gałąź `augmentations`, `docs/ocr_generator_v2.md`.

## TL;DR

- Błędy są **optyczne, nie językowe**: 609 pomyłek wieloznakowych, wśród nich 0 ortograficznych (`rz↔ż`, `ch↔h`, `ó↔u`). Model myli kształty, nie pisownię.
- Największa grupa to **zrośnięte pary liter z kursywy**: `ci→u` (31% wszystkich „ci”), `cz→w` (17%), `rz→n` (10%), `oc→a`, `ls→b` (55%!), `rn→m`. Generator v2 **nie potrafi** ich wytworzyć: renderuje znak po znaku i dodatkowo zawsze rozsuwa litery.
- `k→h` (29% wszystkich „k”) prawie nie zareagował na fonty z pętlowym „k” dodane w v2: 471 (lora_v1) → 373–448 (v2, zależnie od checkpointu).
- Diakrytyki: rozdzielczość nie jest problemem. Duża część `ś→s` / `ż→z` to linie, w których **autor nie postawił** diakrytyku, a etykieta ma poprawną ortografię. Najpierw decyzja o polityce etykiet, potem generowanie.

## 1. Punkt wyjścia: czego v2 nie pokrywa

| zjawisko | liczby (handlabeled, 42 305 znaków) | v2 |
|---|---|---|
| zrośnięte pary kursywy (2:1) | `ci→u` 94, `cz→w` 87, `rz→n` 65, `ci→w` 47, `oc→a` 26, `ls→b` 23, `rn→m` 9 | brak: render per znak + dodatni tracking |
| rozpad litery na dwie (1:2) | `k→lu` 22, `k→li` 7, `d→ol` 16, `b→lo` 8, `ń→ri` 5 | brak: glif zawsze cały, w jednym kawałku |
| podobne kształty 1:1 | `k→h` 439, `a→o` 244, `t→ł` 93, `ł→t` 44, `k→b` 56, `r→n` 55, `g→o` 40 | częściowo (fonty), bez efektu na `k→h` |
| diakrytyki | `ś→s` 143 (37%), `ż→z` 29%, `ą→a` 14%, `ą→a,` 6 | zawsze idealnie nad literą; ablacja `noC_degrad` (bez morfologii): `ś→s` bez zmian |
| wielkość liter | 2.6% edycji, pary o identycznym kształcie (`P/p`, `O/o`, `W/w`) | `caps` jest; dalej nie warto |

Najbardziej mylone znaki łącznie: `k` 45.5%, `ś` 45.7%, `ż` 44.9%, `c` 40.7%, `ą` 34.5%, `z` 24.2%. Zdarzenia są rozłożone na 5–8 z 8 źródeł, więc to ogólne słabości modelu, a nie cecha jednego autora.

## 2. Możliwości generowania

Każdy punkt jako osobna rodzina z flagą `--disable <nazwa>` (tak jak w v2), żeby dało się ją wyłączyć w ablacji.

### 2.1 Łączenia i zrosty liter (`cursive_joins`), priorytet 1

Dlaczego v2 tego nie ma: `char_renderer.render_text_per_char` rysuje każdy znak osobno, odstęp liczy z `font.getlength(char)` (bez kerningu, ligatur i alternatyw kontekstowych OpenType) i zawsze **dodaje** odstęp: `tracking_ratio` 0.06–0.20 rozmiaru fontu, `x_stretch` 1.05–1.25 (`transforms.WordStyle.random`). Fonty kursywne (Playwrite, DancingScript, MarckScript) tracą przez to swoje połączenia, a sąsiednie litery nigdy się nie stykają.

Opcje, od najtańszej:
1. **Ujemny tracking per para**: dla części słów (np. 30%) tracking z zakresu ujemnego (−0.15…0) i losowe dosunięcie wybranych par (`c+i`, `c+z`, `r+z`, `o+c`, `l+s`, `r+n`, `c+o`) tak, żeby glify na siebie zachodziły. Jedna zmiana w `render_text_per_char`, bez nowych zależności.
2. **Render całych słów z kształtowaniem tekstu** dla fontów kursywnych: `ImageFont.truetype(..., layout_engine=ImageFont.Layout.RAQM)` + `features=["liga", "calt", "kern"]`. Font sam łączy litery. Transformacje per znak wtedy odpadają, zostają słowne (elastic, morfologia). Najlepiej jako tryb losowany per linia (np. 30–40% linii w fontach kursywnych).
3. **Syntetyczne kreski łączące**: między glifami rysować krzywą Béziera od punktu wyjścia litery (prawy dolny skraj tuszu na wysokości x) do punktu wejścia następnej. Daje kursywę nawet z fontów drukowanych. Najdroższe, ale najbardziej uniwersalne.
4. **Rozpad glifu (1:2)**: odwrotność zrostu. Losowo przerwać ciągłość litery (wymazać 1–2 px wąski pas w punkcie łączenia brzuszka i laski: `k`, `d`, `b`) albo lekko rozsunąć jej części. Adresuje `k→lu`, `d→ol`, `b→lo`.

Pomiar: P zdarzeń `ci→u`, `cz→w`, `rz→n`, `ls→b`, `k→lu` z `analyze_ngram.py` przed i po.

### 2.2 Ukierunkowana treść: kontrastujące n-gramy (`confusable_ngrams`)

Cel: model ma częściej widzieć **obie strony** mylonej pary w tym samym stylu, np. „ci” i „u”, „cz” i „w”, „k” i „h”, „a” i „o”, „t” i „ł”, „ls” i „b”.

- źródło priorytetów: `results/ngram/multi_events.csv` (P i lift zdarzeń) + `results/ngram/unit_error_stats.csv` (błąd per znak po oczyszczeniu),
- score słowa w puli = Σ po n-gramach słowa P(błędu n-gramu); próbkowanie z wagą `1 + γ·score`, **obcięte do ≤ 3×**, przy czym 50–70% linii zostaje bez zmian,
- dodać słowa „po drugiej stronie” pary (zawierające `u`, `w`, `h`, `n`), żeby model nie przesunął się w odwrotną pomyłkę (`h→k`, `u→ci`, `w→cz`). Monitorować kierunki odwrotne,
- **nie** brać słów z testsetu; ważyć słowa z istniejących pul (anatomia, ICD-11, badania).

Ważne rozróżnienie: oversampling pomaga przy znakach **rzadkich** (`ś` 387, `ż` 136, `ą` 232, `ź` 74 wystąpień w teście). Przy znakach częstych, a źle czytanych (`k` 1526, `c` 1325, `z` 2120), problemem jest **kształt** (2.1, 2.4), nie ekspozycja.

### 2.3 Diakrytyki jako osobne obiekty (`diacritic_placement`)

Renderer per znak daje tu idealny punkt zaczepienia: dla znaku z diakrytykiem można wyznaczyć **maskę diakrytyku** = render(`ś`) − render(`s`) w tym samym foncie i rozmiarze (dla `ł` nie działa, bo to nie diakrytyk, tylko przekreślenie). Z maską można:
- **przesunąć** (dx do ~0.5 szerokości litery, zwykle w prawo; dy w górę), obrócić, przeskalować. W piśmie ręcznym kreska bywa nad następną literą (np. nad „ci” w „kości”),
- **wydłużyć/spłaszczyć** kreskę `ś/ź/ć/ń`, zamienić kropkę `ż` na kreskę poziomą i odwrotnie,
- **oderwać ogonek** `ą/ę` w dół-prawo (`ą→a,` 6 przypadków: ogonek czytany jako przecinek),
- **przesuwać / gubić kropkę nad `i`**: część `ci→u` to brak kropki,
- **pominąć diakrytyk w obrazie przy zachowaniu go w etykiecie**, ale tylko jeśli polityka etykiet jest ortograficzna (patrz 3). Rzędu 10–20% wystąpień. Uczy przywracania z kontekstu słowa, czyli z modelu języka dekodera.

Rozdzielczość: Surya **nie zmniejsza** naszych linii (zmniejsza dopiero powyżej 1024×512 px; 79% linii jest powiększanych). Gubienie diakrytyków nie zależy od wysokości linii (8–15%, tak jak zwykłe litery). Nie trzeba zmieniać skalowania; `phone_photo` z downscale 0.45–0.85 może zostać.

### 2.4 Warianty kształtu pojedynczych liter (`glyph_variants`)

Renderer per znak pozwala podmieniać glif **pojedynczej litery** niezależnie od fontu reszty słowa:
- pula wariantów dla mylonych liter: `k` (pętlowe, otwarte, „h z zawijasem”), `a` (otwarte u góry, czyli jak `o` z ogonkiem), `t` (wysoka poprzeczka jak `ł`), `ł` (kreska nisko/krótko), `z` (kursywne „ʒ”, źródło `cz→w`, `rz→n`), `r` (jak `n` bez prawej laski), `s` (długie „ʃ”, wariant `ś` jednego autora),
- źródło wariantów: glify z innych fontów z puli (podmiana tylko tej litery), albo glify narysowane wektorowo/ręcznie (patrz wnioski poboczne),
- prawdopodobieństwo per litera rzędu 20–40%, spójne w obrębie linii (ten sam „autor” pisze `k` tak samo).

Szczególnie `k→h` (439) i `a→o` (244). Samo dorzucenie fontów dało niewiele, więc trzeba wymusić **częstość** mylącego wariantu, a nie tylko jego obecność w jednym z 35 fontów.

### 2.5 Leksyka domenowa (prior dekodera)

Zdarzenia koncentrują się w terminach anatomicznych: `ci→u` w 32% w „kości”, `ls→b` w „kulszowy”, `cz→w` w „przyczep”, `rz→n` w „powierzchnia”; do tego „przyśrodkowy”, „piszczelowej”. Model zwraca nie-słowa („hosu”, „hubzowa”), więc dekoder nie korzysta jeszcze z kontekstu. Większe pokrycie słownictwa anatomicznego (np. z polskiego mianownictwa anatomicznego, nie z testsetu) daje dekoderowi prior, który rozstrzyga niejednoznaczne kształty. To uzupełnia 2.1–2.4, ich nie zastępuje.

### 2.6 Proporcje, ablacja, ewaluacja

- każda rodzina osobno w ablacji, jak w v2 (`--disable`), na tym samym budżecie co `v2_200k`,
- metryki: globalny CER oraz **P per zdarzenie** z `training/ocr/error_analysis/run_all.sh` (`ci→u`, `cz→w`, `rz→n`, `ls→b`, `k→h`, `a→o`, `ś→s` …) i kierunki odwrotne,
- wyciek: `handlabeled` to zbiór selekcji checkpointu, a wagi z 2.2 też z niego pochodzą. Wagi liczyć na części źródeł (split po zeszytach), oceniać na pozostałych albo na nowym held-oucie,
- zdarzenia z jednego źródła (`ń→ri`, `rw→m`, w dużej mierze `łk→th`) pominąć: to charakter pisma jednej osoby.

### 2.7 Kolejność (szacunek zysku)

| rodzina | adresowane edycje (z 5402) | koszt wdrożenia |
|---|---:|---|
| 2.1 łączenia / zrosty / rozpad | ~600 | niski (ujemny tracking) → wysoki (kreski) |
| 2.4 warianty glifów (`k`, `a`, `t/ł`, `z`) | ~800 (`k→h` 439, `a→o` 244, `t↔ł` 137) | średni |
| 2.3 diakrytyki | ~250, ale część to etykiety | niski–średni |
| 2.2 kontrastujące n-gramy | wspiera 2.1 i 2.4 | niski |
| 2.5 leksyka | pośrednio | niski |

Zmniejszenie adresowanych edycji o połowę daje ok. −1.5 pp CER (12.8% → ~11.3%). To optymistyczne, bez kosztów ubocznych.

## 3. Decyzja przed v3: polityka etykiet dla diakrytyków

Ręczny przegląd losowych 16 + 16 linii z `ś→s` / `ż→z` (ocena wizualna):

| przypadek | `ś→s` | `ż→z` |
|---|---:|---:|
| autor nie postawił diakrytyku, etykieta ortograficzna | ~6 | ~11 |
| `ś` jako osobny glif („ʃ” ze zrośniętą kreską, jeden autor) | ~5 | — |
| diakrytyk jest (często przesunięty), model go pomija | ~4 | ~3 |
| niejasne | ~1 | ~2 |

- **Etykiety ortograficzne** (jak teraz): model ma przywracać diakrytyki z kontekstu. Wtedy 2.3 „pomijanie w obrazie” jest wskazane, a w pracy warto podawać CER także bez tej klasy błędu.
- **Etykiety dyplomatyczne** (tak jak napisano): trzeba przelabelować testset, a „pomijanie w obrazie” byłoby szkodliwe.

Do dokumentów medycznych sensowniejsza wydaje się wersja ortograficzna, ale to decyzja projektowa.

## 4. Wnioski poboczne: fonty

- **35 fontów v2 to za mało różnorodności kształtów tam, gdzie trzeba.** Pętlowe `k` jest w kilku fontach (Playwrite*), a `k→h` się nie zmieniło. Problemem jest częstość wariantu, a nie jego obecność: rozwiązanie w 2.4, a nie kolejne fonty.
- **Fonty kursywne są „spłaszczane” przez renderer** (2.1). Część zalet DancingScript/MarckScript/Playwrite jest dziś niewykorzystana.
- **Diakrytyki w fontach są zawsze idealnie wycentrowane i przyklejone** do litery. Żaden font nie ma przesuniętej kreski, oderwanego ogonka ani „ʃ” zamiast `ś`, stąd 2.3.
- Marek_6, Marek_7, Pokladowski_1: `ł` wygląda jak `T` z zawijasem (wcześniej `ł→T`); Marek_1–8 i Pokladowski_1 nie mają `>` (strzałki odpadają).
- Fonty zmienne (Playwrite, DancingScript, ShantellSans) renderują się tylko w domyślnej grubości; można losować oś `wght` (`font.set_variation_by_axes`).
- **Robienie nowych fontów**: arkusze alfabetu od kilku osób **spoza testsetu** (osobne źródła!) → font przez Calligraphr albo FontForge, z kilkoma wariantami na literę (Calligraphr obsługuje warianty losowane). Łatwy sposób na realne `k`, `a`, `z`, `ś` i diakrytyki. Glifów nie wycinać z `handlabeled`, bo to wyciek.
- **Alternatywa dla fontów**: generatywna synteza pisma (dyfuzyjne modele stylu pisma, rodzina GAN-ów z notatki Pomysły). Daje naturalne łączenia i zrosty, ale jest drogie i ma ryzyko złych etykiet. Do rozważenia dopiero po 2.1–2.4.
