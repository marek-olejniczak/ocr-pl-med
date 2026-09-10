# Generator linii OCR v2 — co zmieniono i dlaczego

Stan na 2026-09-10, gałąź `augmentations`, commity `1f6314a`…`d7d85a4`.

## 1. Punkt wyjścia: wyniki poprzednich treningów

Model: Surya (text_recognition 2025_09_23) + LoRA r=64, α=64, dropout 0.2,
q/v/o_proj, trenowana na `ocr_800k` (40 000 kroków × batch 8 ≈ 0,4 epoki,
lr 2e-5, bf16, RTX 3090). Testset: 2160 linii z prawdziwych notatek
studentów medycyny (zeszyty w kratkę fotografowane telefonem, cropy z detektora).

| Model | CER | WER | Exact match |
|---|---|---|---|
| Surya baza | 19,9% | 62,7% | 13,6% |
| **Surya + LoRA (ocr_800k)** | **15,9%** | **55,8%** | **18,6%** |
| TrOCR-PL fine-tuned | 23,7% | – | – |

Analiza błędów LoRA, która wyznaczyła każdą zmianę poniżej:

| Obserwacja w wynikach | Liczby |
|---|---|
| Najczęstsze podmiany znaków | `k→h` 434 (w 397 liniach), `a→o` 216, `t→ł` 115, `ś→s` 102, `k→b` 50, `c→w` 47, `w→W` 47, `ł→T` 46 |
| Diakrytyki | 13,6% diakrytyków rozpoznanych źle |
| CER wg długości GT | 1–5 znaków **30,8%** (n=199), 6–12 **21,4%** (n=634), 13–25 15,8%, 26–45 14,1%, 46+ 13,7% |
| Udział krótkich linii w teście | 39% linii ma ≤12 znaków |
| CER wg autora | 6,6% (Unaczynienie Miednica) … 31,0% (anatomia_zmysly) |
| Treść testu vs nasze dane | test: cyfry 0,5% znaków, wersaliki 7,1%, 10% linii to nagłówki WIELKIMI, 7% z wypunktowaniem, 4% ze strzałką; nasze `ocr_800k`: cyfry ~22%, wersalików prawie brak |
| Tło i kadr | kratka zeszytowa, zdjęcie telefonem (cień, rozmycie), ogonki sąsiednich linii w kadrze; nasze dane: biały skan, kadr bez sąsiadów |
| Geometria cropów | wysokość mediana 35 px (p10 24, p90 81), szerokość mediana 204 px |
| Artefakt benchmarku | 3 predykcje z `<br>` i halucynowaną drugą linią |

Wniosek: model traci najwięcej na tym, czego w danych treningowych **nie było**:
krótkie pojedyncze słowa, wersaliki, kratka, telefon, sąsiedzi w kadrze,
słownictwo anatomiczne. Generator v2 dodaje dokładnie te elementy.

## 2. Co dodano — rodzina po rodzinie

Każda rodzina ma nazwę, stałą prawdopodobieństwa i flagę `--disable <nazwa>`
w `src/generate_ocr_lines.py`, więc badanie ablacyjne może ją wyłączyć
pojedynczo. `--disable all` odtwarza stary generator (v1, ten od `ocr_800k`).

### Treść (`src/ocr_content.py`)

| Rodzina | Co robi | Parametry | Adresuje |
|---|---|---|---|
| `short_words` | linia = jedno słowo 3–12 znaków (25% szans na dwa słowa); słowa ważone częstością w puli anatomicznej | 34% linii | CER 21–31% na krótkich cropach, 39% testsetu |
| `caps` | nagłówki WIELKIMI LITERAMI; wielka litera na początku frazy | 8% linii nagłówki, 12% fraz z wielką literą | `w→W`, `p→P`; 7% wersalików w teście |
| `arrows_bullets` | prefiksy `- `, `• `, `* `, `1) `, `a) `, `1. `; strzałki `->`, `=>` między terminami (tylko w fontach z glifem `>`) | 8% wypunktowań, 6% strzałek | 7% / 4% linii testsetu |
| `anatomy_vocab` | pule zamiast treści formularzowej; cyfry spadają z ~22% do ~3% znaków | frazy: anatomia ręczna 44%, anatomia LLM 18%, ICD-11 14%, badania lab. 8%, zabiegi 8%, dolegliwości 8% | słownictwo i dekoder językowy modelu |

Rozkład rodzajów linii przy wszystkim włączonym: word 34%, phrase 30%,
heading 8%, bullet 8%, arrow 6%, form_text 8%, number 3%, mix 3%.

Pule słownictwa (`resources/`):

| Plik | Zawartość | Źródło |
|---|---|---|
| `anatomy_seed.txt` | 450 terminów i zwrotów (kości, stawy, więzadła, mięśnie, naczynia, nerwy, narządy, terminy położenia i czynności) | napisane ręcznie |
| `anatomy_phrases.txt` | 1467 linijek w stylu notatek studenta ("pp: kresa chropawa", "zgina udo w stawie biodrowym") | Ollama gemma4 e4b lokalnie, 60 partii × 40 tematów, `src/build_anatomy_pool.py`, czyszczenie znaków i długości; jakość merytoryczna średnia, stąd mniejsza waga |
| `icd11_disease_names.txt` | 14 903 nazw chorób bez kodów | `nlp-ner/data/diseases.csv` |
| `lab_tests.txt` | 7 603 badań laboratoryjnych | `nlp-ner/data/tests.csv` |
| `common_procedures.txt`, `common_conditions.txt` | 98 zabiegów, 96 dolegliwości | `nlp-ner` |

Zasady: frazy przycinane tylko na granicy słów (nigdy "Idiopatyczn"); tekst
sanityzowany względem cmap fontu **przed** pomiarem szerokości, więc etykieta
to dokładnie to, co narysowano; nic z testsetu nie trafia do pul ani do promptu LLM.

### Obraz (`src/line_effects.py`)

| Rodzina | Co robi | Parametry | Adresuje |
|---|---|---|---|
| `neighbour_glyphs` | renderuje prawdziwy tekst sąsiedniej linii tym samym fontem, rozmiarem i stylem ręki, przesunięty tak, że w kadr wchodzą **tylko ogonki** (z góry: `j y g p ą ę`) albo **tylko górne części i diakrytyki** (z dołu: `l t d k ł ó ś ź`); tekst sąsiada w 70% ze słów bogatych w te litery | 40% linii; głębokość 12–45% rozmiaru fontu; 25% szans na obu sąsiadów naraz; tusz sąsiada 0–25 jaśniejszy | ciasne cropy z YOLO na gęstych notatkach (wymaganie z 2026-09-10) |
| `grid_paper` | kratka (70%) lub same linie (30%) w kolorze niebieskim, szarym lub czerwonawym, jedna linia przy linii bazowej pisma | 45% linii; oczko 0,55–1,10 wysokości tekstu; linie 1,10–1,60 | tło zeszytu |
| `morphology` | erozja tuszu jądrem 2×2 zmieszana 50–100% z oryginałem, albo dylatacja 1–2 px | 30% linii | gubione diakrytyki (13,6%), `t↔ł`, cienki długopis vs żelowy |
| `elastic` | `ElasticTransform` (α 0,6–1,6·h, σ 0,18–0,30·h) 60% lub `GridDistortion` (3–6 kroków, 0,08–0,22) 40%, albumentations, brzeg replikowany | 30% linii | wygięta kartka, drżenie ręki (prośba promotora o elastic) |
| `phone_photo` | gradient cienia 10–40% pod losowym kątem, balans bieli ciepły/zimny/neutralny, ekspozycja 0,85–1,08, szum, rozmycie 0,4–1,3 px, motion blur 25%, downscale 0,45–0,85 i powrót 55%, JPEG 55–85 | 40% linii (reszta: stare profile skanera clean/gray/photocopy) | zdjęcie telefonem zamiast skanu |

Kolejność: papier → kratka/linijka → sąsiedzi → tusz linii → morfologia →
elastic → urządzenie (telefon lub skaner). Efekty geometryczne działają na
całym cropie, więc etykieta pozostaje ważna.

### Fonty (`resources/fonts/GoogleFonts_extra/`)

Audyt: arkusz "kakao hobby kłykieć łokieć aorta żyła" dla każdej ręki
(`output/phase1_samples/21_fonty_k_a.png`). Żaden z 15 dotychczasowych fontów
nie ma „k" z pętlą, czyli kształtu mylonego z „h"; w Marek_6, Marek_7 i
Pokladowski_1 „ł" wygląda jak „T" z zawijasem (pasuje do `ł→T` 46).

Pobrano 53 kandydatów z repozytorium Google Fonts (OFL), 37 ma pełne polskie
znaki, wybrano 20 odręcznych: PlaywritePL/CZ/HU/HR/AT (szkolne pismo
elementarzowe, k z pętlą), MarckScript, DancingScript, BadScript, Kalam,
PatrickHand, Mansalva, CaveatBrush, Itim, Mali, Pangolin, Charm, Sriracha,
ShantellSans, ArchitectsDaughter, Courgette. Razem **35 fontów** (było 15).
Fonty Marek_1–8 i Pokladowski_1 nie mają glifu `>`, więc strzałki są w nich
zamieniane na zwykłe frazy.

### Narzędzia

- `src/preview_line_families.py` — arkusze próbek: prawdziwe cropy, baseline,
  każda rodzina wymuszona na 100% przy wyłączonej reszcie, wszystko razem, fonty.
  Wynik w `output/phase1_samples/`.
- `src/build_anatomy_pool.py` — budowa puli fraz z lokalnego Ollama.
- Karta zbioru (`dataset_card.json`) zapisuje `families_enabled`,
  `families_disabled` i wszystkie prawdopodobieństwa, więc każdy zbiór ablacji
  jest samoopisujący się.
- Serwis Surya w benchmarku (gałąź `server`, `ecfe28d`) zdejmuje HTML i tekst
  po `<br>`.
- Testy: `tests/test_ocr_line_families.py` (11 testów), razem 102 w repo.

## 3. Czego NIE zmieniono (żeby porównanie było uczciwe)

- Model, LoRA, budżet treningu (40 000 kroków × 8), lr, testset.
- Rozmiar fontu 26–40 px, zakres marginesów kadru, stare profile skanera,
  stara treść formularzowa (nadal 14% linii: form_text/number/mix).
- Generator całych dokumentów (`generate_yolo_dataset.py`) — nietknięty;
  detektor linii działa dobrze.

## 4. Jak z tego korzystać

```
# nowy zbiór, wszystko włączone (odpowiednik ocr_800k, ale v2)
python src/generate_ocr_lines.py --output-dir output/ocr_800k_v2 --count 800000 --seed 2029 --dataset-name ocr_800k_v2

# stary generator (baseline do ablacji)
python src/generate_ocr_lines.py --output-dir output/ocr_800k_v1 --count 800000 --seed 2029 --disable all

# ablacja: bez jednej rodziny
python src/generate_ocr_lines.py --output-dir output/abl_no_neighbour --count 800000 --seed 2029 --disable neighbour_glyphs

# arkusze próbek
python src/preview_line_families.py --real-dataset "C:/Users/tomek/Desktop/ocr-main/dataset"
```

Rodziny: `short_words caps arrows_bullets anatomy_vocab neighbour_glyphs grid_paper morphology elastic phone_photo`.

## 5. Otwarte

- Trening na v2 i pomiar wobec 15,9% CER (faza 2) — odłożony, decyzja z 2026-09-10.
- Aktualny testset 2160 linii z serwera (kopia w DVC ma 1630).
- Pula LLM: rozważyć filtr słownikiem polskim albo lepszy model, jeśli okaże
  się, że zmyślone terminy szkodzą.
- Fonty zmienne (Playwrite, DancingScript, ShantellSans) renderują się tylko
  w domyślnej grubości; można dodać losowanie osi `wght`.
