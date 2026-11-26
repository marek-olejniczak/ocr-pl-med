# Postprocessing OCR: eksperymenty i prace (do pisania pracy)

Stan: 2026-11-26. Kod i liczby: `postprocessing/` (spis plików z liczbami: `postprocessing/results/README.md`).
Przegląd literatury: [[Postprocessing_literatura]]. Wątki całej pracy: [[Watki_pracy_dyplomowej]].

> **Liczby są przejściowe.** Wejdzie finalna wersja zbioru testowego i wszystko liczymy na niej od nowa
> (`postprocessing/experiments/rebuild.py`, instrukcja w README obok). W pracy podajemy liczby z tamtego przebiegu.
> Poniższe pokazują, co sprawdziliśmy i jakie były wnioski.

---

## 1. W pięciu zdaniach

1. Korektor to krok **po** OCR: słowo spoza słownika zamienia na słowo słownikowe, korzystając z modelu pomyłek
   nauczonego na błędach naszej Suryi i z częstości słów w korpusie.
2. Wariant docelowy dodatkowo pyta Suryę, który z 5 kandydatów najlepiej pasuje do **obrazu** linii.
3. Na test/real/v1.2 z Suryą v4: CER 0.1098 → **0.0858** (−2.40 p.p.) z oceną Suryi, → 0.0939 (−1.59 p.p.) bez niej.
4. Trzy czwarte zysku korektora słownikowego pochodzi z wyuczonych pomyłek, a nie z samego słownika.
5. Im lepszy model OCR, tym mniej zostaje do poprawienia; na kartach pacjenta korektor prawie nic nie daje.

**Zasada z 2026-11-26:** wariantem docelowym jest ocena Suryi. W każdym porównaniu w pracy wspominamy też
wariant statyczny (top-5 bez Suryi).

## 2. Pojęcia (do rozdziału o metodyce)

| pojęcie | znaczenie |
|---|---|
| CER / WER | błędy znaków / słów, mikro po wszystkich liniach; normalizacja tylko białych znaków, zgodnie z `eval_cer.py` i benchmarkiem |
| CER pomocniczy | bez wielkości liter i interpunkcji; tylko w raportach |
| ΔCER p.p. | różnica CER w punktach procentowych; „względnie” = ΔCER / CER przed |
| harm | odsetek tokenów poprawnych przed korektą, które korektor zepsuł; limit roboczy **0.3% na walidacji** |
| fixed / broken / changed-wrong | token naprawiony / poprawny zepsuty / błędny zamieniony na inny błędny |
| LOGO | zagnieżdżone leave-one-group-out: dla każdej grupy model pomyłek i progi uczone bez niej |
| dev → test | model i progi tylko z dev, ocena na odłożonych grupach test |
| grupa | źródło notatek (przybliżenie autora; **liczba autorów nieznana**) |
| CI po liniach / po grupach | bootstrap parowany; po grupach szerszy i uczciwszy (linie z jednego źródła są skorelowane) |
| szum strojenia | ok. 0.1 p.p. CER: mniejsze różnice między wariantami **nie są wynikiem** |

## 3. Jak działa korektor (do rozdziału o metodzie)

1. **Czyszczenie linii:** NFC, usunięcie znaków zerowej szerokości i podwójnych spacji, zwijanie zapętleń modelu
   (`deloop`: fraza do 3 słów powtórzona ≥ 3 razy z rzędu, powtórzone segmenty `<br>`).
2. **Które słowa ruszać:** tylko słowa z samych liter, spoza słownika, o długości ≥ `L_min`, nienależące do klas
   chronionych (symbole, liczby, skróty, krótkie wielkie, słowa z dywizem) ani do listy terminów anatomicznych.
3. **Słownik:** PoliMorf 2026-10-18 (słowa znane i kandydaci) + lista 4102 terminów anatomicznych z Wikidata
   (en, la, pl; **tylko chroni**, nie jest kandydatem).
4. **Częstość słowa P(w):** unigramy NKJP z wygładzaniem Simple Good-Turing.
5. **Model pomyłek P(OCR | słowo):** model kanału Brill & Moore (2000), uczony hard-EM na parach (etykieta, wynik
   Suryi): zamiany, wstawienia, usunięcia i jednostki do 2 znaków (`cz→w`, `ci→u`). Wkład jednej grupy w liczniki
   jest ograniczony do 50%.
6. **Kandydaci:** wszystkie słowa w odległości Levenshteina ≤ 3 (rapidfuzz, w partii) oraz warianty różniące się
   tylko polskimi znakami.
7. **Decyzja:** ocena `s = log P(OCR|c) + λ·log P(c)`; zmiana tylko, gdy przewaga nad oryginałem ≥ `δ`
   i nad drugim kandydatem ≥ `μ`. Progi stroimy: najmniej błędów przy harm ≤ limit.
8. **Ocena przez Suryę (wariant docelowy):** dla 5 najlepszych kandydatów liczymy log P(linia z kandydatem | obraz)
   w trybie teacher forcing (ta sama ścieżka co strata w treningu LoRA) i dodajemy `w·(różnica względem oryginału)`.

Wariantów statycznych są **dwa**, nie mylić ich:

| nazwa | kandydaci | rola |
|---|---|---|
| „pełna lista” (opcja 1 z początku prac) | wszyscy | większość ablacji i porównań modeli OCR |
| „top-5 bez Suryi” (`w = 0`) | 5 najlepszych | statyczny punkt odniesienia dla oceny Suryi i tryb zapasowy w paczce |

Top-5 bez Suryi jest lepsze od pełnej listy (−1.59 vs −1.22 p.p.): sama redukcja listy kandydatów pomaga.
Dlatego zysk **samej oceny Suryi** to różnica 2.40 − 1.59 = **0.81 p.p.**, a nie 2.40 − 1.22.

## 4. Dane

| zbiór | linie w ocenie | grupy | kiedy używany |
|---|---|---|---|
| handlabeled przed poprawką GT | 2160 | 8 | P0–P5, pierwsze ablacje (predykcje z 2026-10-01) |
| dataset-v1 (`test/real/v1`) | 2160 | 8 | wynik v2_200k, ablacje modelu pomyłek i listy terminów |
| `test/real/v1.1` | 2210 (+68 `[BLANK]`) | 8 | runda 2 (v3) |
| `test/real/v1.2` | **2904** (+68 `[BLANK]`) | **11** | v4, RysOCR, ocena Suryi |

- v1.2 = v1.1 + 80 stron (541 linii): karty pacjenta i recepta (141 linii, G9), „Nerw Twarzowy” (188, G10),
  „Żyły Głębokie” (212, G11).
- Dev/test w v1.2 (z manifestu): test = noga, anatomia_zmysly, miednica (1069 linii), dev = 1835 linii.
- Linie `[BLANK]` są w zbiorze celowo (testy detekcji); w ocenie OCR je pomijamy.
- **Poprawka etykiet (2026-11-04):** z listy 101 podejrzanych tokenów ręczny przegląd wskazał 39 błędów etykiet
  (38 diakrytyków i jedna literówka); 62 zapisy są wierne obrazowi i zostały. Autorka notatek często pomija kreskę
  nad „ś”. Poprawki są w dataset-v1.

**Uwaga na trzy różne „baseline” v2:** 0.12769 (stary przebieg, stare GT), 0.12727 (stary przebieg, poprawione GT),
0.12701 (nowy przebieg na dataset-v1). Dwa przebiegi tego samego adaptera różniły się w 640 z 2160 linii.
Nie porównywać CER między wierszami o różnych baseline'ach, tylko delty z tym samym protokołem.

## 5. Który wynik cytować

Dziś (do zastąpienia finalnym przebiegiem), Surya `v4_200k_25k`, v1.2, harm ≤ 0.3% na walidacji:

| wariant | LOGO CER (Δ p.p.) | 95% CI po grupach | WER | fixed / broken / ch-wrong | harm | test CER (Δ) | harm test |
|---|---|---|---|---|---|---|---|
| bez korekty | 0.1098 | — | 0.3989 | — | — | 0.1040 | — |
| **top-5 + ocena Suryi** | **0.0858 (−2.40)** | [−2.76, −2.01] | 0.2841 | 1132 / 24 / 387 | 0.40% | 0.0784 (−2.56) | 0.31% |
| top-5 bez Suryi | 0.0939 (−1.59) | [−1.86, −1.25] | 0.3120 | 858 / 19 / 341 | 0.32% | 0.0887 (−1.53) | 0.20% |
| pełna lista, bez Suryi | 0.0975 (−1.23) | [−1.45, −0.96] | 0.3293 | — | 0.27% | 0.0956 (−0.84) | 0.05% |

- **Liczba główna: LOGO.** Test przy ocenie Suryi nie był ściśle jednorazowy (E10).
- Pełna lista: `results/v12/p4_summary.csv` ma −1.23 p.p. (węższa siatka `δ`); po poszerzeniu siatki wychodzi
  −1.22 (`results/v12/README.md`). Różnica w granicach szumu. Finalny przebieg da jedną liczbę.

## 6. Eksperymenty

Oznaczenia odtwarzalności: 🟢 liczy `rebuild.py` (również na nowym zbiorze) · 🟡 jest kod w repo, uruchomienie
ręczne · 🔴 jednorazowe, kodu nie ma w repo (nie przeliczy się na finalnym zbiorze).

### E1. Ile w ogóle da się zyskać? (oracle) 🟢 `ablations`
- **Pytanie:** jaki jest sufit korekty słowo po słowie?
- **Dane:** v2_200k, handlabeled przed poprawką (CER 0.12769). **Metoda:** błędne tokeny podmieniane na etykietę,
  jeśli spełniają warunek wariantu.
- **Wyniki:** każdy błędny token −10.22 p.p.; spoza słownika, z etykietą w słowniku i Lev ≤ 3: −6.70; Lev ≤ 2: −4.75;
  tylko błędy diakrytyków: −0.36; sama higiena tekstu: 0.00.
- **Wnioski:** korekta słownikowa ma duży zapas; samo przywracanie diakrytyków niewiele da; 20% błędnych tokenów
  to poprawne słowa słownika (real-word), których reguła „tylko spoza słownika” nie ruszy.
- **Pliki:** `results/dataset-v1/ORACLE.md`, `oracle*.csv`.

### E2. Słownik, częstości, klasy chronione 🟢 `ablations`
- Pokrycie słów etykiet: PoliMorf 93.27%, z sjp 94.35%; jako kandydaci 89.55%.
- Klasy chronione są w dużej mierze nadmiarowe wobec reguły „same litery, spoza słownika, ≥ 3 znaki”.
- Opcja 0 (higiena) nie zmienia żadnej linii v2: CER bez zmian.
- **Pliki:** `results/dataset-v1/P2_P3.md`, `p2_*.csv`.

### E3. Czy model pomyłek Brill & Moore jest lepszy od zwykłego Levenshteina? 🟢 `ablations`
- Log-likelihood par (etykieta, OCR) na wyłączonej grupie: lepszy na **8 z 8** grup, o 2.3–5.5 nat na błędną parę.
- Jednostki wieloznakowe pomagają na każdej grupie. Najczęstsze: `ls→b`, `ci→u`, `cz→w`, `rn→m`, `rz→n`.
- Źródłem pomysłu była analiza pomyłek `training/ocr/error_analysis/`.
- **Pliki:** `p3_loglik.csv`, `p3_inventory.csv`.

### E4. Pełny protokół i limit szkód 🟢 `static`, `ablations` (`harm_05`, `harm_10`)
- Zagnieżdżone LOGO, siatka progów `λ, d_max, f_min, δ, μ, L_min`.
- Limit przenosi się na test z optymizmem: progi na granicy 0.5% dały 0.81% (stare dane), 0.3% dało 0.40%.
  Stąd wybór **0.3%**.
- v4, tylko dev: limit 0.3% → −0.97 p.p. (harm 0.29%, changed-wrong 117); 0.5% → −1.25 (0.47%, 189);
  1% → −1.46 (0.54%, 309). Błędne podmiany rosną szybciej niż naprawy. **Wybór limitu to decyzja, nie wynik.**
- Test odporności (podział gruby na 6 grup): zysk CER podobny, ale harm 1.40% (v2): progi z notatek ręcznych nie
  przenoszą się w pełni na zrzuty ekranu.

### E5. Co daje zysk? Ablacje modelu pomyłek 🟢 `ablations`
v2_200k, dataset-v1, LOGO, harm ≤ 0.3%. **Najważniejsza tabela o metodzie.**

| wariant | CER | ΔCER p.p. | fixed / broken / ch-wrong | precision | harm |
|---|---|---|---|---|---|
| pełny model | 0.09874 | −2.83 | 778 / 10 / 144 | 0.835 | 0.31% |
| bez limitu udziału grupy | 0.09959 | −2.74 | 757 / 9 / 158 | 0.819 | 0.28% |
| tylko pomyłki 1:1 | 0.10446 | −2.26 | 732 / 12 / 188 | 0.785 | 0.37% |
| jednostajny Levenshtein | 0.11999 | −0.70 | 315 / 8 / 121 | 0.709 | 0.25% |

- Słownik + zwykły Levenshtein: −0.70; wyuczone 1:1: dalsze −1.55; jednostki wieloznakowe: dalsze −0.57.
- Limit udziału grupy: w szumie; zostaje jako ochrona przed pomyłkami jednego zeszytu.
- Usunięta waga `w_diac` dla par diakrytycznych: wynik w szumie, prior jest jednostajny.

### E6. Lista terminów anatomicznych 🟢 `ablations` (`no_domain`, `domain_la_pl`, `domain_in_tuning`)

| LOGO, dataset-v1 | CER | ch-wrong | harm |
|---|---|---|---|
| bez listy | 0.09914 | 144 | 0.71% |
| la, pl | 0.09892 | 144 | 0.56% |
| **en, la, pl** | **0.09874** | 144 | **0.31%** |
| lista także przy strojeniu progów | 0.09472 | **258** | 0.50% |

- Lista działa jako **ochrona**, nie korekta (−0.04 p.p., ale 13 zepsuć mniej).
- Strojenie z listą **odrzucone mimo niższego CER**: +114 błędnych podmian, np. `śluzowa→służbowe`. Płynny błąd
  w tekście medycznym jest gorszy niż widoczny śmieć. Dobry materiał do dyskusji, dlaczego sam CER nie wystarcza.
- Lista pochodzi z zewnętrznego źródła, więc nie ma przecieku z etykiet.

### E7. Starsze ablacje (stare predykcje i GT, punkt odniesienia −2.29 p.p.) 🟢 `ablations`
| wariant | ΔCER | wniosek |
|---|---|---|
| bez klas chronionych P2–P8 | −2.29 | bez różnicy |
| tylko przywracanie diakrytyków | −0.09 | wchłonięte przez pełny korektor |
| + słownik sjp | −2.13 | większy słownik **szkodzi** (harm 0.62%) |
| forward zamiast Viterbi | −2.32 | w szumie, większy harm → Viterbi |

W pracy nie zestawiać tych liczb bezpośrednio z tabelą E5 (inny baseline). Finalny przebieg policzy wszystkie
ablacje na jednym zbiorze.

### E8. Lepszy model OCR a zysk korektora 🟢 `static` (`compare_models`)
v1.2, LOGO, pełna lista kandydatów:

| model OCR | CER przed → po | Δ p.p. | względnie | harm |
|---|---|---|---|---|
| v2_200k | 0.1550 → 0.1297 | −2.53 | −16.3% | 0.30% |
| v3_200k | 0.1446 → 0.1200 | −2.47 | −17.0% | 0.23% |
| v4_200k (10k kroków) | 0.1160 → 0.1025 | −1.35 | −11.6% | 0.20% |
| **v4_200k_25k** | 0.1098 → 0.0975 | −1.23 | −11.2% | 0.27% |
| RysOCR | 0.3249 → 0.3021 | −2.28 | −7.0% | 0.06% |

- v4 bez korekty (0.1098) jest lepszy niż v3 z korektorem (0.1200).
- Na v4 zysk jest o połowę mniejszy: 41% słów, które korektor naprawiał po v3, v4 czyta sam; strojenie wybiera
  `L_min = 6` (krótkich słów nie opłaca się ruszać). Liczba „41%” pochodzi z jednorazowego skryptu 🔴.
- Lepszy OCR i korektor częściowo naprawiają **te same** błędy.
- Surowy CER wszystkich 7 modeli: `results/v12/raw_ocr.csv`.

### E9. Porównanie z RysOCR 🟡
- RysOCR = PaddleOCR-VL + LoRA `kacperwikiel/RysOCR`, wrapper z benchmarku w venv na serwerze, batch 1, prompt `OCR:`
  (na 200 liniach dev CER 0.52 vs 0.63 dla innego promptu). Wrapper wymagał dwóch obejść
  (`results/v12/README.md`): **opisać to w pracy**, bo wpływa na uczciwość porównania.

| zbiór | RysOCR | Surya v4_200k_25k |
|---|---|---|
| v1.2 całość | 0.3249 | **0.1098** |
| v1.2 test | 0.2811 | **0.1040** |
| karty pacjenta | 1.190 | **0.493** |
| RxHandBD test (bez wielkości liter i interpunkcji) | **0.4775** | 0.5816 |
| htrlex core2k (jw.) | 0.1955 | **0.0662** |

- Uczciwie: na RxHandBD (angielskie nazwy leków) RysOCR jest lepszy od każdej Suryi, a v4 nie poprawił się wobec v2.
- RysOCR zapętla się na części linii; po `deloop` na v1.2: 0.3042.

### E10. Ocena kandydatów przez Suryę (wariant docelowy) 🟢 `candidates`, krok GPU, `rescore`
- **Pytanie:** czy model OCR, patrząc na obraz, wybierze kandydata lepiej niż sam model pomyłek?
- **Obserwacja:** Surya prawie zawsze daje własnemu odczytowi wyższe log P niż etykiecie (dekodowanie zachłanne),
  więc **sama pewność modelu nie mówi, czy poprawiać**. Ale wśród kandydatów wybiera dobrze: gdy etykieta jest
  w top-5 (1589 błędnych tokenów), Surya stawia ją na 1. miejscu 1462 razy, model pomyłek 1343 razy; od oryginału
  woli ją tylko 93 razy. Dlatego potrzebne jest połączenie obu ocen.
- **Wynik:** tabela w §5.
- Wdrożeniowy `SuryaScorer` sprawdzony z eksperymentem: mediana różnicy log P 5e-7, maks. 0.55 w 1% tekstów
  (bf16 w partiach), zero zmian decyzji.
- **Zastrzeżenia (obowiązkowo w pracy):**
  1. Strojono na trzech coraz szerszych siatkach, a każdy przebieg liczył też test → test **nie jest ściśle
     jednorazowy**; główną liczbą jest LOGO.
  2. Optimum leży na brzegu siatki: `δ = −16` w 5 z 11 foldów; w protokole dev → test wybrane `w = 2` to maksimum
     siatki (paczka wdrożeniowa ma `w = 1`). Zysk może być niedoszacowany.
  3. Harm w LOGO 0.40% przy celu 0.3% ustawionym na walidacji.
  4. Przy strojeniu wewnętrznym kandydaci pochodzą z modelu pomyłek foldu zewnętrznego (mały przeciek do wyboru
     kandydatów, nie do progów).
- Koszt: ok. 7 wariantów na linię z kandydatami (4.8 na linię strony), jeden przebieg modelu bez dekodowania;
  RTX 3090 ok. 100 wariantów/s (pomiar przed przyspieszeniem z §10).

### E11. Dekodowanie Suryi: lookahead i zadanie 🔴
- Lookahead (przyjmowanie kilku tokenów naraz) jest **wyłączony w configu modelu** (`multi_output_distance = 0`),
  wymuszenie progu nie zmienia żadnej linii → hipoteza o nietrenowanych głowicach odpada.
- Inferencja jest deterministyczna przy tej samej kolejności i batchu.
- `ocr_with_boxes` (zadanie z treningu LoRA) vs `ocr_without_boxes` (benchmark): −0.07 p.p., CI95 [−0.31, +0.10]
  → bez różnicy.
- Skrypty tylko w `.scratch/postprocessing/logprobs/` (poza gitem).

### E12. Zbiory zewnętrzne i karty pacjenta 🔴
- htrlex core2k: korektor 0.0662 → 0.0590 (−0.72 p.p.; 96 naprawionych, 15 zepsutych). Etykiety upstream są
  małymi literami, więc ocena bez wielkości liter.
- RxHandBD test: 0.5816 → 0.5752, **całość z `deloop`**; sama korekta: 20 zmian, 0 napraw (angielskie leki).
- Surya v3 zapętlała się na RxHandBD: CER 0.710 → 0.600 po `deloop` (pomiar z rundy 2, z wielkością liter i interpunkcją,
  dlatego liczby różnią się od tabeli w E9).
- Karty pacjenta (G9): CER ok. 0.5 w każdym modelu. W 6 liniach ramka łapie nadruk formularza, który OCR czyta:
  225 z 598 edycji (0.493 → 0.350 bez tych linii). Kilka etykiet jest niewiernych obrazowi; bez nich ok. 0.33.
  Z błędnych słów 82 ze 102 jest w PoliMorf, więc **słownik nie jest wąskim gardłem**.
- 41 podejrzanych etykiet nowych stron v1.2 czeka na przegląd (`.scratch/postprocessing/review/suspect_gt_v12_new.csv`).
  **Sprawdzić przed finalnym zbiorem.**

## 7. Wyniki negatywne (też warto opisać)

Wszystkie z tej tabeli to 🔴 (kod usunięty z repo) albo wynik z fazy badań; nie przeliczą się na finalnym zbiorze.
Podawać z datą i zbiorem, na którym je zmierzono.

| pomysł | zbiór / model | wynik | decyzja |
|---|---|---|---|
| model pomyłek zależny od pozycji litery w słowie | dataset-v1, v2 i v3 | v2 istotnie gorzej (+0.18…+0.35 p.p.), v3 nieistotnie lepiej (−0.10…−0.18) | odrzucone |
| osobne progi dla Tytułowych / WIELKICH | v1, v3 | −0.12 p.p., CI [−0.45, +0.12], progi niestabilne | odrzucone |
| ICD-11 jako słownik anatomiczny | pokrycie słów spoza słownika | za mało (12–32 z 273) | potrzebna odmiana przymiotników |
| listy PESEL (imiona, nazwiska) | notatki | rzadkie nazwiska chronią śmieci OCR (`mozna`, `sredni`) | odrzucone w tej formie |
| lista leków RPL jako kandydaci | syntetyka medyczna | nazwy leków 0.36 → 0.42, CER bez zmian; **wynik optymistyczny** (ten sam model szumu co w korektorze) | niezweryfikowane na prawdziwych danych; na G9 nieopłacalne |
| NER (HerBERT, `ner-medical-pl`) jako ochrona | syntetyka i notatki | F1 encji: etykiety 0.79 → OCR 0.69 → po korekcie 0.71; na notatkach 2.6× więcej fałszywych PERSON | odradzane |

**Syntetyczny zbiór medyczny** (ok. 63 tys. linii, szum z modelu pomyłek): korektor −2.9 p.p., harm 0.24%; nazwiska
0.23 → 0.59, leki 0.30 → 0.36, choroby 0.27 → 0.48. Zepsucia to głównie skróty medyczne WIELKIMI (`TURP→TURĄ`,
`FSH→FSM`). Szum pochodzi z tego samego modelu co korektor, więc to **górne oszacowanie**, nie wynik.

**Nie sprawdzaliśmy** (tylko przegląd literatury): n-gramy słów / LM kontekstowy (błędy real-word), polski leksykon
anatomiczny z odmianą jako kandydaci, LLM jako ranker, fine-tuning ByT5, ROVER.

## 8. Ograniczenia (do rozdziału o zagrożeniach dla wiarygodności)

1. **Grupa to źródło, nie autor.** Jeśli dwa źródła pisała jedna osoba, LOGO przecieka styl pisma.
2. **Mało grup** (8 albo 11): przedziały po grupach są szerokie i niestabilne.
3. **Projekt korektora powstał po analizie całych danych** (siatka progów, klasy chronione, pomysł listy terminów).
   Zamrożony test chroni przed przeciekiem progów i modelu pomyłek, ale nie projektu.
4. **Wybór checkpointu:** v2_200k wybrano na handlabeled (`--cer-callback`, `training/ocr/surya/README.md`),
   więc jego bezwzględny CER jest in-sample. Dla v4 **trzeba ustalić u osoby trenującej**, na czym wybierano
   checkpoint. Delta korektora dotyczy obu stron porównania jednakowo, bezwzględny CER nie.
5. **Etykiety są wierne obrazowi:** część „zepsuć” to dodane diakrytyki, które autorka pominęła
   (w wyniku v1: 7 z 10 zepsutych).
6. **Harm nie liczy błędnych podmian.** Zawsze podawać też changed-wrong.
7. **Domena:** prawie same notatki anatomiczne. Wniosków nie przenosić na recepty i dokumentację kliniczną.
8. Strojenie używa przybliżenia (zmiana liczby błędów przy podmianie jednego tokenu); wynik końcowy liczony dokładnie.

## 9. Finalny przebieg: lista kontrolna

1. Dostać: finalny zbiór (`crops.csv`, `manifest.csv` z kolumną `split`), przebieg benchmarku na nim, adapter.
2. Dopisać reguły grup dla nowych źródeł (`experiments/data/group_rules.json`); **grupa = źródło/autor**.
3. Przed uruchomieniem ustalić i nie zmieniać po obejrzeniu wyników: limit harm (0.3%), listę języków
   (`en,la,pl`), siatki, które modele porównujemy.
4. `rebuild.py` (instrukcja: `postprocessing/experiments/README.md`): `prepare static candidates`, krok GPU,
   `rescore bundle ablations report`.
5. Sprawdzić w logach, czy optimum nie leży na brzegu siatki (`δ`, `w`). Jeśli poszerzamy siatkę, opisać to
   i traktować test jako użyty więcej niż raz.
6. Zastąpić liczby w §1, §5, §6 i w `postprocessing/package/README.md` §3.

**Do rozważenia (dziś niezaimplementowane):** ocena obecnej paczki (zbudowanej na v1.2) bez żadnego przeuczania
na stronach, które pojawią się dopiero w finalnym zbiorze. To byłby jedyny wynik w pełni spoza danych, na których
powstawał korektor. Trzeba ją zrobić **przed** zbudowaniem nowej paczki i tylko na nowych stronach.

## 10. Wdrożenie i szybkość (do rozdziału implementacyjnego)

- **Pakiet `ocr-postprocess`** (`postprocessing/package/`): `pip install`, jedna komenda `setup`, API w Pythonie,
  przykład podłączenia do workera Suryi, serwis HTTP (tylko statyczny), test zgodności sprzętu `check-surya`.
  Sama aplikacja nie jest częścią tych prac.
- **Skompilowany słownik:** start korektora 22.3 s → 1.5 s, strona 30 linii na CPU 3.4 → 1.5 s, RAM 2.0 → 1.6 GB.
  Budowa słownika: 5.6 s, częstości: 1.6 s, indeks kandydatów: 15.9 s; teraz liczone raz. Wyniki identyczne
  na 2904 liniach.
- **Enkoder obrazu raz na linię** (2026-11-26): wcześniej ten sam wycinek był kodowany osobno dla każdego wariantu.
  Na GTX 1650: 1.85× szybciej (41 linii, 391 wariantów), różnica ocen maks. 0.06 nat, ten sam zwycięzca w 41/41.
- **Przenośność:** GTX 1650 (fp16) i CPU (float32) przechodzą test zgodności; względem RTX 3090 (bf16) różnica
  ocen maks. 0.52 nat. Mac (MPS) i Windows niesprawdzone. Tabela czasów: README pakietu, §4.
- Model pomyłek w paczce zapisany jako JSON (wcześniej pickle).

## 11. Chronologia

| data | co |
|---|---|
| 2026-10-29 | przegląd literatury i pomiar kandydatów |
| 2026-11-04 | decyzje o zakresie; P0–P5: ewaluacja, oracle, słownik, model pomyłek, korektor, ablacje; poprawka 39 etykiet |
| 2026-11-05/06 | dataset-v1 jako punkt odniesienia, lista terminów, ablacje modelu pomyłek |
| 2026-11-15/16 | runda 2: pozycja w słowie, wielkość liter, syntetyka medyczna, listy, NER, `deloop`, v3 na v1.1 |
| 2026-11-24 | v4 na v1.2, porównanie z RysOCR, zbiory zewnętrzne |
| 2026-11-25 | limit harm na v4, karty pacjenta, ocena kandydatów przez Suryę, paczka wdrożeniowa |
| 2026-11-26 | skompilowany słownik (start 23 s → 1.5 s), pakiet `ocr-postprocess`, `rebuild.py` |
