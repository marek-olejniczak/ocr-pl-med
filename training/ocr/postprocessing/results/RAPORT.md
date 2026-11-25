# Statyczny postprocessing wyjścia Surya: raport (P0–P5, stan 2026-11-06)

Korektor słownikowy z modelem kanału (opcja 1, z opcją 2 wchłoniętą) na wyjściu Surya v2_200k
(LoRA r64), bez ponownej inferencji. Dane: handlabeled z dataset-v1, 2160 linii, 8 grup (źródeł notatek).
Pełna tabela: [`TABLE.md`](TABLE.md); szczegóły faz: [`ORACLE.md`](ORACLE.md), [`P2_P3.md`](P2_P3.md),
[`P4.md`](P4.md). Raport nie zawiera tekstu linii.

> **Aktualizacja 2026-11-24/25.** Raport opisuje Suryę v2_200k na dataset-v1. Stan na Surya v4 (generator z ~50 fontami)
> i test/real/v1.2 (2904 linie, 11 grup): [`v12/README.md`](v12/README.md). Korektor daje tam −1.23 p.p. w LOGO
> (0.1098 → 0.0975) i −0.84 p.p. na zamrożonym teście, mniej więcej połowę zysku z v2/v3.

## 1. Wynik główny

Wariant główny (decyzje 2026-11-04 do 2026-11-06): limit szkód harm ≤ 0.3% poprawnych tokenów na
walidacji wewnętrznej, progi strojone bez listy terminów, a przy korekcie działa lista TA `en,la,pl`
chroniąca terminy (§4b). Dwa protokoły: zamrożony test (progi i model kanału tylko z dev, grupy G3,
G5–G8) oraz zagnieżdżone leave-one-group-out na wszystkich liniach.

| | zamrożony test (948 linii) | LOGO (2160 linii) |
|---|---|---|
| CER (jak `eval_cer.py`) | 0.12377 → **0.10621** | 0.12701 → **0.09874** |
| ΔCER | −1.76 p.p. (−14.2%) | −2.83 p.p. (−22.3%) |
| 95% CI ΔCER, po grupach | [−2.89, −0.27] p.p. (3 grupy) | [−3.38, −1.89] p.p. |
| 95% CI ΔCER, po liniach | [−2.05, −1.44] p.p. | [−3.04, −2.61] p.p. |
| WER | 0.4477 → 0.3625 | 0.4831 → 0.3556 |
| tokeny fixed / broken / changed-wrong | 181 / 0 / 37 | 778 / 10 / 144 |
| precision / recall | 0.830 / 0.196 | 0.835 / 0.278 |
| harm | 0.00% | 0.31% |

Do pracy idzie zamrożony test; LOGO uczy model i progi na większej liczbie grup i obejmuje też zrzuty
ekranu, więc pokazuje szerszy obraz. CER pomocniczy w LOGO (casefold, bez interpunkcji): 0.11741 → 0.08780.
Linie lepsze / gorsze w LOGO: 665 / 30. Zysk jest na każdej z 8 grup: od −0.27 p.p. (G4, baseline 0.048)
i −1.46 (G2, baseline 0.218) do −3.77 p.p. (G3).

## 2. Ablacje (wszystkie przy harm ≤ 0.3%, podział drobny)

Liczone na predykcjach z 2026-10-01 i GT przed poprawką z §4a, więc ich punktem odniesienia jest
tutejszy wiersz −2.29 p.p., a nie wynik z §1. Ablacje modelu kanału na dataset-v1: §2a, listy terminów: §4b.

| wariant | ΔCER p.p. | harm test | wniosek |
|---|---|---|---|
| opcja 1 (główny) | −2.29 | 0.40% | |
| bez jednostek wieloznakowych (`cz→w`, `ci→u`, `ls→b`…) | −1.66 | 0.50% | jednostki wieloznakowe dają **0.6 p.p.**, czyli ok. 28% zysku |
| bez klas chronionych P2–P8 | −2.29 | 0.40% | prawie bez różnicy (3 zmiany); patrz §3 |
| opcja 2 osobno (tylko warianty diakrytyczne) | −0.09 | 0.40% | małe, zgodnie z pułapem oracle −0.36 |
| + sjp „odmiany” | −2.13 | 0.62% | większy słownik szkodzi: więcej śmieciowych kandydatów |
| forward zamiast Viterbi | −2.32 | 0.59% | ΔCER w granicach szumu, większy harm: zostaje Viterbi |
| harm ≤ 0.5% zamiast 0.3% | −2.82 | 0.81% | +0.5 p.p. CER kosztem 2× harm |
| podział gruby (6 grup) | −2.50 | 1.40% | zysk CER odporny; harm nie (§3) |

Opcja 0 (NFC, zero-width, spacje): 0 zmienionych linii, predykcje są już czyste. Zostaje tylko
jako metryka pomocnicza.

## 2a. Ablacje modelu kanału na dataset-v1 (2026-11-06)

LOGO na 2160 liniach, harm ≤ 0.3%, lista `en,la,pl`, kod końcowy. Każdy wariant ma progi strojone od nowa.

| wariant | CER | ΔCER p.p. | fixed / broken / changed-wrong | precision | harm |
|---|---|---|---|---|---|
| **pełny model (główny)** | **0.09874** | −2.83 | 778 / 10 / 144 | 0.835 | 0.31% |
| bez limitu udziału grupy (`--no-group-cap`) | 0.09959 | −2.74 | 757 / 9 / 158 | 0.819 | 0.28% |
| tylko pomyłki 1:1 (`--no-multi`) | 0.10446 | −2.26 | 732 / 12 / 188 | 0.785 | 0.37% |
| jednostajny Levenshtein (`--uniform`) | 0.11999 | −0.70 | 315 / 8 / 121 | 0.709 | 0.25% |

- **Wyuczone pomyłki dają trzy czwarte zysku.** Sam słownik ze zwykłą odległością to −0.70 p.p. Pomyłki 1:1
  dokładają 1.55 p.p., a jednostki wieloznakowe 0.57 p.p., czyli 20% zysku. Bez jednostek ubywa 46 poprawek i
  przybywa 44 błędnych podmian, więc poprawiają one głównie trafność (precision 0.785 → 0.835).
- **Limit udziału grupy:** różnica 0.09 p.p. i 14 błędnych podmian mieści się w szumie strojenia (niżej), więc
  wpływ jest mały. Limit zostaje, bo chroni model przed pomyłkami jednego zeszytu, co sprawdza LOGO tylko częściowo.
- **Usunięta waga `w_diac`.** Model miał wagę 20 dla par z tej samej rodziny diakrytycznej, ale działała ona tylko
  na pary już widziane. Para niewidziana dostawała wartość domyślną, taką samą jak każda inna (`z→ż` = `z→q`).
  Z wagą: LOGO 0.09862, 780 / 10 / 143, test 0.10642, 179 / 0 / 37. Bez wagi: 0.09874 i 0.10621, czyli różnica
  w granicach szumu. Prior w `P(b|a)` jest teraz jednostajny, a wszystkie wiersze dataset-v1 przeliczono bez wagi.
- **Jawnego podobieństwa kształtów liter nie ma.** Podobieństwo wchodzi przez dane, bo model uczy się pomyłek Suryi.
  Prior z podobieństwa glifów miałby mały pułap: na wyłączonych grupach pary 1:1 niewidziane w uczeniu to 159 z 4862
  operacji błędu (3.3%), z czego 4 diakrytyczne. Liczba może być zaniżona, bo wyrównanie omija drogie pary.
- **Szum strojenia to około 0.1 p.p.** Usunięcie wagi zmieniło wynik główny o 0.01 p.p., ale w wariantach blisko
  granicy harm progi przeskoczyły: wariant z listą w strojeniu z 0.09602 na 0.09472, a wariant 1:1 z 691 na 732
  poprawki przy prawie tym samym CER. Różnic rzędu 0.1 p.p. między wariantami nie należy traktować jako mocnych.

## 3. Obserwacje

Obserwacje w §3 i §4 pochodzą z tego samego starszego przebiegu co §2, poza progami do wdrożenia.

- **Model kanału jest kluczowy.** Na każdej wyłączonej grupie log-likelihood par (GT, pred)
  pod modelem Brill & Moore jest wyższy niż pod jednostajnym Levenshteinem o 2.3–5.5 nat/parę
  (P3), a jednostki wieloznakowe są w ablacji najważniejszym składnikiem.
- **Klasy chronione są w dużej mierze redundantne** z regułą kwalifikacji: korygujemy tylko
  rdzenie z samych liter (dywiz wewnątrz dopuszczalny) o długości ≥ 3 spoza słownika, więc
  cyfry, symbole i tokeny 1–2 literowe (P1–P3, P7) odpadają wcześniej. Faktycznie działają P4/P5
  (skróty z kropką, lista skrótów) i P8 (dywiz), na kilku tokenach.
- **Limit szkód przenosi się z optymizmem.** Progi wybierane na granicy limitu na walidacji
  dają na teście ok. 1.3–1.6× większy harm. Dlatego wariant główny ma limit 0.3%.
- **Test odporności na podziale grubym**: zysk CER podobny (−2.50 p.p.), ale harm 1.40%, prawie
  w całości z jednego foldu: test na połączonej grupie zrzuty+dataN (869 linii) przy progach
  strojonych na 5 pozostałych źródłach (μ = 0.5, δ = −4) dał harm 2.9%. Pozostałe foldy ≤ 0.54%.
  Progi strojone na „zeszytach” nie przenoszą się w pełni na zrzuty ekranu: przy wdrożeniu na
  nowym typie materiału trzeba je zweryfikować.
- **Prefiltr Lev ≤ 3 wystarcza**: GT jest wśród kandydatów dla 79% kwalifikujących się błędnych
  tokenów; słów GT poza zasięgiem, które wygrałyby z wybranym kandydatem, jest 3.5%.
- **Parametry**: λ ≈ 0.75–1.0, d_max = 3, μ = 2–3 są stabilne między foldami; δ i f_min się wahają.
  Progi do wdrożenia (dataset-v1, LOGO na wszystkich 8 grupach, harm ≤ 0.3%, bez listy):
  `deploy_params.json` (λ = 0.75, δ = −3.0, μ = 3.0, d_max = 3, L_min = 3, f_min = 1).

## 4. Co psuje korektor

Wariant główny „psuje” 13 tokenów, które były zgodne z GT:

| typ | liczba | przykłady (pred → korekta) |
|---|---|---|
| GT bez diakrytyku, korektor dodał poprawny diakrytyk | 11 | `mięsnie→mięśnie` (×2), `mozna→można`, `posladkowa→pośladkowa`, `krązka→krążka`, `MÓŻDZEK→MÓŻDŻEK` |
| prawdopodobna literówka w GT | 1 | `kłaciek→kłaczek` |
| faktyczne zepsucie | 1 | `groore→grosze` |

Po odjęciu przypadków z błędną etykietą realny harm to ok. 1–2 tokeny na ok. 3200 poprawnych
(≈ 0.05%). Wszystkie 11 przypadków diakrytycznych są na liście podejrzanych etykiet z P1.

Przy łagodniejszym limicie (harm ≤ 0.5%, 26 zepsutych) dochodzą terminy angielskie i łacińskie
spoza słownika polskiego: `femur→temu` (×4), `condyle→londynie` (×2), `knee→nie`, `medius→medium`,
`minimus→minimum`, `inferior→interior`, `tibial→tibia`, oraz rzadkie polskie terminy
(`zasłaniacz→zasłaniasz`). To bezpośredni argument za opcją 4 (leksykon dziedzinowy).

Zmiany „na inny błąd” (117): ok. 1/3 to tokeny, które po korekcie różnią się od GT tylko
wielkością litery lub interpunkcją (`Prewód→Przewód` przy GT `przewód`); reszta to podmiana na
inne słowo, często bliższe GT (`hubzowa→kulszowa` przy GT `kulszową`), ale nie zawsze.

## 4a. Wynik na GT po ręcznym przeglądzie (2026-11-04)

Przegląd 101 podejrzanych tokenów GT (`suspect_gt_diacritics.csv`, kolumny `gt_token_fix`, `decision`):
39 to błędy etykiety (38 diakrytyków + literówka `kuklowate→kulkowate`), 62 jest wiernych obrazowi. Lista zmian w formacie DagsHub (`dataset/annotations.csv`) przekazana do aktualizacji repozytorium danych.
Autorka notatek (zwłaszcza zrzuty G7/G8) często pomija kreskę nad ś; GT zapisuje to, co jest na
obrazku, nie poprawną polszczyznę. Poprawki naniesione na `benchmark/dane/handlabeled/labels.csv`
i `data/baseline/predictions.jsonl` (wielkość liter z oryginału). Ablacje z §2 liczone na GT sprzed poprawki.

| | baseline | opcja 1 (główny, harm ≤ 0.3%) |
|---|---|---|
| CER | 0.12727 (było 0.12769) | **0.09989** (−2.74 p.p., −21.5%) |
| 95% CI ΔCER, grupy / linie | | [−3.34, −1.81] / [−2.94, −2.52] p.p. |
| WER | 0.4794 | 0.3556 |
| fixed / broken / changed-wrong / net | | 763 / 17 / 163 / +746 |
| precision / recall / harm test | | 0.809 / 0.273 / **0.53%** |

Model kanału uczony na poprawionych parach śmielej dodaje diakrytyki, więc zysk rośnie (−2.29 → −2.74 p.p.),
ale harm przekracza cel (0.40% → 0.53%), prawie w całości w G7/G8 (1.1–1.2%). 17 zepsutych tokenów:

| typ | liczba | przykłady |
|---|---|---|
| diakrytyk dodany tam, gdzie autorka go pominęła (GT wierny obrazowi) | 8 | `mięsnie/mięsnia→mięś…` (×4), `posladkowa/-ej→poś…` (×2), `mozna→można`, `sredni→średni` |
| termin angielski/łaciński poprawiony na polskie słowo | 8 | `femur→temu` (×4), `knee→nie`, `inferior→interior`, `tibial→tibia`, `minimus→minimum` |
| literówka w GT | 1 | `kłaciek→kłaczek` |

Pierwsza grupa to „szkoda” tylko względem transkrypcji wiernej obrazowi; druga to faktyczne zepsucia,
które adresuje opcja 4.

## 4b. dataset-v1 i lista terminów anatomicznych (2026-11-05)

Predykcje: ten sam adapter v2_200k na `dataset/test/real/v1` (`data/baseline/`), CER 0.12701, WER 0.48307.
GT w v1 zawiera poprawki z §4a. Grupy G1, G2, G4 to dokładnie test z `benchmark/splits`.

**Lista TA:** 4102 słowa spoza PoliMorf z etykiet Wikidata dla elementów z identyfikatorem TA98/TA2
(CC0; `data/domain/ta_terms.tsv`, generator `evaluation/build_domain.py`). Źródło jest zewnętrzne, więc
nie ma przecieku z GT. Lista tylko chroni terminy przed „poprawieniem”; jej słowa nie są kandydatami korekty.
Języki wybiera `--domain` (w API `domain_langs`), domyślnie lista jest wyłączona. Progi stroimy bez niej,
więc zmiana języków nie wymaga ponownego strojenia.

| LOGO, 2160 linii | CER | fixed / broken / changed-wrong | precision | harm |
|---|---|---|---|---|
| bez listy | 0.09914 | 778 / 23 / 144 | 0.823 | 0.71% |
| `la,pl` | 0.09892 | 778 / 18 / 144 | 0.828 | 0.56% |
| **`en,la,pl` (główny)** | **0.09874** | 778 / 10 / 144 | 0.835 | **0.31%** |
| `en,la,pl`, lista także w strojeniu | 0.09472 | 914 / 16 / 258 | 0.769 | 0.50% |

- **Lista działa jak ochrona, nie jak korekta:** −0.04 p.p. CER, 13 zepsuć mniej, bez kosztu. Korzyść daje
  głównie angielski: słowa tylko angielskie to 67 z 5591 słów, 64 z nich w zrzutach G7/G8. Poza zrzutami harm
  wynosi 0.11% niezależnie od listy, a na zamrożonym teście wynik z listą i bez niej jest identyczny.
- **Dlaczego bez strojenia z listą:** lista zwalnia budżet harm i strojenie luzuje próg δ. Daje to niższy
  CER (na teście 0.10408 zamiast 0.10621), ale +136 poprawek kosztuje +114 błędnych podmian w LOGO, np.
  `dotyku→bałtyku` (przez `błotyku`) albo `śluzowa→służbowe`. Widoczny śmieć z OCR staje się płynnym, błędnym
  słowem. Harm tego nie liczy, a CER prawie nie karze, a w tekście medycznym to gorsze niż widoczny błąd.
- **Co zostało zepsute (10 tokenów w wariancie głównym):** 7 to diakrytyki pominięte przez autorkę
  (`mięsnie→mięśnie`, `mozna→można`), czyli szkoda tylko względem transkrypcji wiernej obrazowi.
- **Harm to kilka–kilkanaście tokenów**, więc różnice rzędu 0.1 p.p. nie są mocne.

Zastrzeżenie: linie testowe nie brały udziału w strojeniu ani w uczeniu, ale projekt korektora
(siatka progów, klasy chronione, pomysł listy) powstał po analizie wszystkich 2160 linii. Test oceniono też ponownie
po usunięciu `w_diac` (§2a), bo zmienił się kod.

## 5. Ograniczenia (do pracy)

- **Fold = źródło, nie autor.** Liczba autorów handlabeled jest nieznana („kilka osób”); jeśli
  dwa źródła pisała ta sama osoba, LOGO przecieka styl pisma. Podział gruby (6 grup) daje
  podobny zysk CER, więc efekt przecieku na ΔCER jest mały; na harm już nie.
- **8 grup**: przedział po grupach jest szeroki i niestabilny (bootstrap z 8 jednostek);
  podajemy też przedział po liniach, który jest optymistyczny (linie z jednego źródła są skorelowane).
- **Checkpoint in-sample**: v2_200k wybrano na tym samym zbiorze, więc bezwzględny CER baseline
  jest optymistyczny; delta korekty dotyczy obu stron porównania jednakowo.
- **Błędy etykiet GT**: 39 z 101 podejrzanych tokenów poprawiono (§4a) i te poprawki są w dataset-v1.
  Pozostałe 62 są wierne obrazowi; część „zepsuć” to diakrytyki pominięte przez autorkę.
- **Szum inferencji:** dwa przebiegi tego samego adaptera (2026-10-01 i dataset-v1) różnią się w 640 z 2160
  linii, przy CER 0.12727 i 0.12701 na tym samym GT. Różnice CER rzędu 0.03 p.p. mogą być szumem.
- **Błędne podmiany (changed-wrong) nie są ograniczane:** strojenie pilnuje tylko harm. To 144 tokeny
  w LOGO, a część z nich zamienia śmieć z OCR w wiarygodnie wyglądające, błędne słowo (§4b).
- Strojenie progów używa przybliżenia (zmiana edycji przy podmianie jednego tokenu); wynik
  testowy liczony dokładnie.
- Nie ruszamy błędów real-word (20% błędnych tokenów to formy istniejące w słowniku) ani
  sklejeń/rozbić tokenów: to opcja 3 (n-gramy słów, ticket 11).

## 6. Dalej

- **Ograniczenie błędnych podmian**: warunek na changed-wrong lub precyzję obok harm przy strojeniu, na dev.
- **Opcja 4 (ticket 12)**: lista TA chroni terminy (§4b). Otwarte: polskie terminy jako `target`
  (wymaga fleksji, lista ma głównie mianowniki).
- **Logprobs Surya, lookahead, ocena kandydatów przez model**: wymaga stacji z GPU.
- **Opcja 3 (ticket 11)**: kontekst słów dla real-word i decyzji przy małym marginesie.
- Ostrożniejsza siatka μ (np. μ ≥ 2) jako sposób na odporność harm; to wymaga osobnego
  uzasadnienia, bo wybór po obejrzeniu testu byłby przeciekiem.
