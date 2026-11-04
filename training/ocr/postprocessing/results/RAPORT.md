# Statyczny postprocessing wyjścia Surya: raport (P0–P5, 2026-11-04)

Korektor słownikowy z modelem kanału (opcja 1, z opcją 2 wchłoniętą) na wyjściu Surya v2_200k
(LoRA r64), bez ponownej inferencji. Test: handlabeled, 2160 linii, 8 grup (źródeł notatek).
Pełna tabela: [`TABLE.md`](TABLE.md); szczegóły faz: [`ORACLE.md`](ORACLE.md), [`P2_P3.md`](P2_P3.md),
[`P4.md`](P4.md). Raport nie zawiera tekstu linii.

## 1. Wynik główny

Wariant główny: limit szkód harm ≤ 0.3% poprawnych tokenów na walidacji wewnętrznej
(decyzja 2026-11-04). Predykcje out-of-fold z zagnieżdżonego leave-one-group-out.

| | baseline | opcja 1 (główny) |
|---|---|---|
| CER (jak `eval_cer.py`) | 0.12769 | **0.10483** (−2.29 p.p., −17.9%) |
| 95% CI ΔCER, bootstrap po grupach / po liniach | | [−2.93, −1.48] / [−2.47, −2.09] p.p. |
| CER pomocniczy (casefold, bez interpunkcji) | 0.11953 | 0.09554 |
| WER | 0.4817 | 0.3773 |
| tokeny fixed / broken / changed-wrong / net | | 642 / 13 / 117 / +629 |
| precision / recall korekt | | 0.832 / 0.228 |
| harm na teście | | **0.40%** (cel ≤ 0.5% spełniony) |
| linie lepsze / gorsze (test znaków) | | p < 1e-6 |

To ok. 48% pułapu oracle „token OOV → GT, GT w słowniku, lev ≤ 2” (−4.75 p.p.) i 34% pułapu
dla lev ≤ 3 (−6.70 p.p.). Zysk jest na każdej z 8 grup (aneks w `TABLE.md`): od −0.27 p.p.
(G4, baseline już 0.045) i −0.86 (G2, najtrudniejsze pismo, baseline 0.236) do −3.66 p.p. (G3).

## 2. Ablacje (wszystkie przy harm ≤ 0.3%, podział drobny)

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

## 3. Obserwacje

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
  Progi do wdrożenia (LOGO na wszystkich 8 grupach, harm ≤ 0.3%): `deploy_params.json`
  (λ = 1.0, δ = 1.0, μ = 3.0, d_max = 3, L_min = 3, f_min = 20).

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

## 5. Ograniczenia (do pracy)

- **Fold = źródło, nie autor.** Liczba autorów handlabeled jest nieznana („kilka osób”); jeśli
  dwa źródła pisała ta sama osoba, LOGO przecieka styl pisma. Podział gruby (6 grup) daje
  podobny zysk CER, więc efekt przecieku na ΔCER jest mały; na harm już nie.
- **8 grup**: przedział po grupach jest szeroki i niestabilny (bootstrap z 8 jednostek);
  podajemy też przedział po liniach, który jest optymistyczny (linie z jednego źródła są skorelowane).
- **Checkpoint in-sample**: v2_200k wybrano na tym samym zbiorze, więc bezwzględny CER baseline
  jest optymistyczny; delta korekty dotyczy obu stron porównania jednakowo.
- **Błędy etykiet GT**: 101 tokenów GT (1.8% słów) to formy spoza słownika, których wariant
  z diakrytykiem jest w słowniku; w 26 przypadkach Surya przewidział wariant z diakrytykiem.
  Oficjalny GT bez zmian; część „zepsuć” korektora to w rzeczywistości poprawki (§4). Lista
  czeka na ręczny przegląd (`suspect_gt_diacritics.csv`, kolumna `decision`).
- **Baseline 0.12769** pochodzi z predykcji z 2026-10-01 (`eval_cer.py`); wartość 0.127243
  z ablacji checkpointów to inny przebieg inferencji (przypis w pracy).
- Strojenie progów używa przybliżenia (zmiana edycji przy podmianie jednego tokenu); wynik
  testowy liczony dokładnie.
- Nie ruszamy błędów real-word (20% błędnych tokenów to formy istniejące w słowniku) ani
  sklejeń/rozbić tokenów: to opcja 3 (n-gramy słów, ticket 11).

## 6. Dalej

- **Opcja 4 (leksykon dziedzinowy, ticket 12)**: łacina i angielski anatomiczny jako `known`
  (ochrona) i ewentualnie `target`. Bezpośrednio adresuje część zepsuć z §4.
- **Opcja 3 (ticket 11)**: kontekst słów dla real-word i decyzji przy małym marginesie.
- Po przeglądzie listy podejrzanych etykiet: druga wersja wyniku na GT z poprawionymi diakrytykami.
- Ostrożniejsza siatka μ (np. μ ≥ 2) jako sposób na odporność harm; to wymaga osobnego
  uzasadnienia, bo wybór po obejrzeniu testu byłby przeciekiem.
