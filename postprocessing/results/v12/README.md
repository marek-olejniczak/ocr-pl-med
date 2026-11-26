# Surya v4 i test/real/v1.2 (2026-11-24)

Predykcje: benchmark serwera `ocr-v4_20261123_140539` (7 modeli na real_v1.2), import `evaluation.import_bench`.
v1.2 = v1.1 (etykiety bez zmian) + 80 stron `new` (541 linii): karty pacjenta 1–3 i recepta (141 linii, G9_karta),
„Nerw Twarzowy” (188, G10), „Żyły Głębokie” (212, G11). 68 linii `[BLANK]` zostaje w zbiorze dla detekcji, w ocenie OCR
je pomijamy (`metrics.load_predictions`): 2904 linie. Dev/test z manifest.csv = TEST_GROUPS z `run_holdout`
(noga, anatomia_zmysly, miednica): dev 1835, test 1069.

## Surowy OCR (`raw_ocr.csv`)

| model | całość | dev | test | karty (G9) | nowe notatki (G10+G11) |
|---|---|---|---|---|---|
| surya (bez LoRA) | 0.2251 | 0.2208 | 0.2347 | 0.524 | 0.202 |
| ocr800k | 0.1991 | 0.1942 | 0.2101 | 0.587 | 0.155 |
| v2_200k | 0.1550 | 0.1566 | 0.1513 | 0.608 | 0.116 |
| v3_base_200k | 0.1499 | 0.1519 | 0.1454 | 0.507 | 0.115 |
| v3_200k | 0.1446 | 0.1465 | 0.1405 | 0.526 | 0.112 |
| v4_200k (10k kroków) | 0.1160 | 0.1190 | 0.1091 | 0.592 | 0.080 |
| v4_200k_25k | **0.1098** | 0.1124 | 0.1040 | 0.493 | 0.074 |

## Korektor, wariant główny (harm ≤ 0.3%, lista TA `en,la,pl`)

LOGO po 11 grupach (`p4_v12_*`, `p4_summary.csv`) i protokół dev → test (`holdout_v12_*`).

| model | LOGO | Δ | harm | fixed / broken / changed_wrong | test (progi z dev) | Δ test | harm test |
|---|---|---|---|---|---|---|---|
| v2_200k | 0.1550 → 0.1297 | −2.53 p.p. (−16.3%) | 0.30% | 1176 / 15 / 241 | 0.1513 → 0.1311 | −2.03 | 0.06% |
| v3_200k | 0.1446 → 0.1200 | −2.47 (−17.0%) | 0.23% | 1140 / 12 / 255 | 0.1405 → 0.1204 | −2.01 | 0.00% |
| v4_200k | 0.1160 → 0.1025 | −1.35 (−11.6%) | 0.20% | 751 / 12 / 220 | 0.1091 → 0.0981 | −1.10 | 0.00% |
| v4_200k_25k | 0.1098 → **0.0975** | −1.23 (−11.2%) | 0.27% | 688 / 16 / 185 | 0.1040 → **0.0956** | −0.84 | 0.05% |

CI po grupach dla v4_200k_25k: [−1.45, −0.96] p.p.

Wnioski:
- v4_200k_25k bez korekty (0.1098) jest lepszy niż v3 z korektorem (0.1200). Korektor dalej daje istotny zysk,
  ale mniej więcej o połowę mniejszy niż na v2/v3.
- Zasięg: złych tokenów kwalifikujących się do korekty 2453 (v3: 3292), GT wśród kandydatów 1797 (v3: 2458),
  naprawione 688, czyli 38% osiągalnych (v3: 46%). 41% tokenów naprawianych przez korektor na v3 v4 czyta sam.
- Strojenie wybiera prawie zawsze `L_min = 6` (na v2/v3 3–4): na v4 korekta krótkich słów przestała się opłacać.
  Progi do wdrożenia dla v4: `deploy_params.json` (`lam 0.75, delta −4, mu 3, d_max 3, L_min 6, f_min 0`).
- Profil złych tokenów v4 jak w v3: ok. 81% to inne błędy, 6% tylko diakrytyki, 5% interpunkcja, 1% wielkość liter.
- Karty pacjenta (G9): CER ok. 0.5 we wszystkich modelach, korektor prawie bez efektu (GT wśród kandydatów
  dla 31 z 80 złych tokenów): krótkie pola, nazwiska, daty, kody ICD, nazwy leków. W kilku liniach predykcja zawiera
  nadruk formularza, którego nie ma w etykiecie: możliwy rozjazd wycinka i etykiety.

Odtworzenie całości (również ocena Suryi i paczka): `python rebuild.py configs/v12.json …`, opis w
`../../experiments/README.md`. Pliki w tym katalogu mają dawne tagi `v12_<model>_ta`; `rebuild.py` używa nazw modeli.

## Porównanie z RysOCR (2026-11-24)

RysOCR = PaddleOCR-VL + LoRA `kacperwikiel/RysOCR` (apache-2.0), benchmarkowy wrapper `rysocr_wrapper.py` uruchomiony
w procesie na serwerze (venv na /home; obrazu Dockera nie da się zbudować, bo partycja z Dockerem jest pełna), batch 1,
max_new_tokens 128, prompt `OCR:` z karty modelu: na 200 liniach dev CER 0.52 wobec 0.63 dla `Transcribe the text.`
Dwie uwagi do wrappera: podmiana `create_causal_mask` psuje obecną rewizję PaddleOCR-VL (dae2996, brak `cache_position`;
wyłączona w skrypcie uruchomieniowym), a ścieżka wsadowa ucina prompt po `attention_mask.sum()`, co przy lewym
dopełnieniu jest błędne (dlatego batch 1).

CER (WER w nawiasie); RxHandBD i htrlex bez wielkości liter i interpunkcji (`normalize_aux`), RxHandBD tylko split test bez flag.

| model | v1.2 całość | v1.2 test | karty (G9) | nowe notatki | RxHandBD test (1111) | htrlex core2k (2000) |
|---|---|---|---|---|---|---|
| RysOCR | 0.3249 (0.796) | 0.2811 | 1.190 | 0.255 | **0.4775** (1.013) | 0.1955 (0.799) |
| Surya v2_200k | 0.1550 (0.514) | 0.1513 | 0.608 | 0.116 | 0.5412 (0.945) | 0.0726 (0.300) |
| Surya v3_200k | 0.1446 (0.495) | 0.1405 | 0.526 | 0.112 | 0.6256 (1.141) | 0.0723 (0.309) |
| Surya v4_200k | 0.1160 (0.409) | 0.1091 | 0.592 | 0.080 | 0.5600 (0.926) | 0.0694 (0.301) |
| Surya v4_200k_25k | **0.1098** (0.399) | **0.1040** | **0.493** | **0.074** | 0.5816 (1.004) | **0.0662** (0.287) |

Po `deloop` (ta sama higiena dla wszystkich): RysOCR v1.2 0.3042, RxHandBD bez zmian; Surya v3 RxHandBD 0.5319, v4_25k 0.5738.
RysOCR zapętla się na części linii notatek („1. 1. 1. …”): 10 najgorszych z 200 linii pilota to 38% edycji.

Korektor (ten sam protokół) na wyjściu RysOCR: LOGO 0.3249 → 0.3021 (−2.28 p.p., harm 0.06%, changed_wrong 594),
dev → test 0.2811 → 0.2624 (−1.87 p.p.). RysOCR z korektą jest prawie 3× gorszy niż Surya v4 bez korekty.
Na RxHandBD (angielskie nazwy leków z recept z Bangladeszu) RysOCR jest lepszy od każdej Suryi, a v4 nie poprawił się względem v2.

## Dalsze pomiary (2026-11-25, lokalnie)

**Limit harm na v4 (tylko dev, LOGO wewnątrz dev, test nieczytany).** harm ≤ 0.3%: 0.1124 → 0.1027 (−0.97 p.p., harm 0.29%,
changed_wrong 117); ≤ 0.5%: → 0.0999 (−1.25, harm 0.47%, changed_wrong 189); ≤ 1%: → 0.0978 (−1.46, harm 0.54%,
changed_wrong 309). Każde +0.2 p.p. harm daje ok. −0.3 p.p. CER, ale błędne podmiany rosną szybciej niż naprawy.
`L_min = 6` zostaje w 13 z 16 foldów niezależnie od limitu: krótkie słowa nie opłacają się w żadnym wariancie.
Wybór budżetu harm to decyzja, nie wynik; wariant główny zostaje przy 0.3%.

**Zbiory zewnętrzne** (model kanału z całego v1.2, progi `deploy_params.json`, lista TA `en,la,pl`, CER bez wielkości liter):
htrlex core2k 0.0662 → 0.0590 (−0.72 p.p.; 119 zmian: 96 naprawionych, 15 zepsutych); RxHandBD test 0.5816 → 0.5752,
całość z higieny (`deloop`), sama korekta 20 zmian i 0 naprawionych: angielskie nazwy leków spoza słownika.

**Karty pacjenta (G9).** Wycinki pasują do etykiet, ale w 6 liniach bbox łapie nadruk formularza, który OCR czyta:
225 z 598 edycji G9 (CER 0.493 → 0.350 bez tych linii). Kilka etykiet jest niewiernych obrazowi (np. `11.05.2024` przy
`11.05.24`, `tablet` przy `tabl.`, `Jzo` przy `J20.`); bez nich ok. 0.33. Złe tokeny G9: 148, z czego 46 to liczby/kody/krótkie
skróty; z 102 słów 82 mają GT w PoliMorf, 8 w RPL (6× `tabl`, Plavix, Forxiga), 1 w listach imion. Słownik nie jest
wąskim gardłem: GT wypada poza kandydatów (odległość > 3) albo słowo jest krótsze niż `L_min`. Lista RPL nie jest warta wdrożenia;
`tabl` lepiej jako skrót chroniony. Na nowych notatkach (G10+G11) 225/239 złych słów ma GT w PoliMorf; brakujące to
terminy anatomiczne (`zażuchwowy`) i literówki w GT.

## Ticket 20: logprobs Suryi i ocena kandydatów (2026-11-25, serwer RTX 3090)

**Krok A (inferencja, `eval_cer.py` na real_v1.2, v4_200k_25k).** Powtórka daje identyczne linie (inferencja deterministyczna
przy tej samej kolejności i batchu). `FOUNDATION_MULTI_TOKEN_MIN_CONFIDENCE=1.01` nie zmienia żadnej linii: w configu modelu
`multi_output_distance = 0`, więc lookahead jest wyłączony i hipoteza o nieuczonych głowicach odpada. Zadanie `ocr_with_boxes`
(na nim uczono LoRA) zamiast `ocr_without_boxes` (benchmark): 358 linii inaczej, ΔCER −0.07 p.p., CI95 [−0.31, +0.10], 144 lepsze
/ 147 gorsze: bez różnicy. `eval_cer.py` liczy 0.10790 przy 0.10981 z benchmarku: 6 linii różnych (zapętlenia).

**Krok B (log P z teacher forcingu).** Surya daje własnemu odczytowi prawie zawsze wyższe log P niż etykiecie (dekodowanie
zachłanne), więc sama pewność nie mówi, czy poprawiać.

**Krok C (ocena kandydatów).** Dla tokenów kwalifikujących się do korekty 5 kandydatów (wg ch + 0.75·lp, model kanału bez grupy
linii), dla każdego log P(linia z kandydatem | obraz) − log P(oryginał | obraz) = sd. Decyzja: s(c) = ch + lam·lp + w·sd,
progi i w strojone w zagnieżdżonym LOGO (`evaluation/run_rescore.py`, kandydaci `rescore_cands.py`, serwer `surya_lp.py`).
Wśród 1589 złych tokenów z GT w top-5 Surya stawia GT na 1. miejscu 1462 razy (model kanału 1343), ale od oryginału woli GT tylko 93 razy.

| wariant (harm ≤ 0.3% na walidacji wewnętrznej) | LOGO 11 grup | zamrożony test | changed_wrong (LOGO) |
|---|---|---|---|
| korektor dotychczasowy (pełna lista kandydatów) | 0.1098 → 0.0976 (−1.22 p.p.) | 0.1040 → 0.0956 (−0.84) | 198 |
| top-5 kandydatów, bez Suryi (w = 0) | → 0.0939 (−1.59), CI linie [−1.72, −1.45] | → 0.0887 (−1.53) | 341 |
| top-5 + ocena Suryi | → **0.0858 (−2.40)**, CI linie [−2.55, −2.24] | → **0.0784 (−2.56)** | 387 |

(`rescore_w0.json`, `rescore_surya.json`). Zastrzeżenia: (1) warianty z oceną Suryi strojono na trzech coraz szerszych siatkach
i każdy przebieg liczył też test, więc test nie jest ściśle jednorazowy; główną liczbą jest LOGO. (2) W najszerszej siatce
`delta` dalej stoi na brzegu (−16) w 5 z 11 foldów i w = 2 to maksimum siatki: zysk może być nieco większy. (3) Harm w LOGO
0.40% (cel 0.3% liczony na walidacji wewnętrznej). (4) Kandydaci dla strojenia wewnętrznego pochodzą z modelu kanału foldu
zewnętrznego (mały przeciek grupy testowej do wyboru kandydatów, nie do progów). Siatkę `delta` w `run_cv` poszerzono do −16:
wariant główny bez zmian (−1.22 vs −1.23), progi `deploy_params.json` przeliczone (delta −4.5).

Wdrożenie: paczka `v4_200k_25k` w pakiecie (tryb statyczny top-5 i tryb z oceną Suryi), opis w `../../package/README.md`.
