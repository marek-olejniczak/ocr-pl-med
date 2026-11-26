# Postprocessing OCR: co przetestowano i co działa

Stan 2026-11-25 (wcześniej 2026-11-16), branch `postprocessing`. Kod: `training/ocr/postprocessing/`, pełny raport z tabelami:
`results/RAPORT.md`, `results/TABLE.md`.

Statyczny korektor wyjścia Surya bez ponownej inferencji: słownik (PoliMorf, częstości NKJP), model kanału
Brill & Moore uczony na parach (GT, predykcja), decyzja z progami strojonymi pod limit szkód. Ocena w zagnieżdżonym
leave-one-group-out (LOGO, grupy = źródła notatek) i na zamrożonym teście.

**Harm** = odsetek tokenów poprawnych przed korektą, które korektor zepsuł. Limit roboczy: harm ≤ 0.3%.
**Szum strojenia** to ok. 0.1 p.p. CER: mniejsze różnice między wariantami nie są wynikiem.

## Wynik główny

| dane | model OCR | CER przed → po | ΔCER | harm |
|---|---|---|---|---|
| dataset-v1, zamrożony test (948 linii) | v2_200k | 0.1238 → 0.1062 | −1.76 p.p. (CI [−2.05, −1.44]) | 0.00% |
| dataset-v1, LOGO (2160 linii) | v2_200k | 0.1270 → 0.0987 | −2.83 p.p. (CI grup [−3.38, −1.89]) | 0.31% |
| dataset-v1, LOGO | v3_200k (pilot) | 0.1196 → 0.0980 | −2.16 p.p. | 0.42% |
| test/real v1.1, LOGO (2210 linii, bez `[BLANK]`) | v3_200k | 0.1280 → **0.1029** | −2.51 p.p. (CI grup [−3.02, −1.60]) | 0.32% |
| test/real v1.1, LOGO | v2_200k | 0.1351 → 0.1073 | −2.78 p.p. | 0.34% |
| syntetyczny tekst medyczny (63 tys. linii) | szum v3 | 0.1199 → 0.0908 | −2.9 p.p. | 0.24% |
| test/real v1.2, LOGO 11 grup (2904 linie, bez `[BLANK]`) | **v4_200k_25k** | 0.1098 → **0.0975** | −1.23 p.p. (CI grup [−1.45, −0.96]) | 0.27% |
| test/real v1.2, zamrożony test (1069 linii) | v4_200k_25k | 0.1040 → 0.0956 | −0.84 p.p. | 0.05% |
| test/real v1.2, LOGO | v3_200k | 0.1446 → 0.1200 | −2.47 p.p. | 0.23% |
| test/real v1.2, LOGO | RysOCR (konkurencja) | 0.3249 → 0.3021 | −2.28 p.p. | 0.06% |
| test/real v1.2, LOGO, top-5 + ocena Suryi | v4_200k_25k | 0.1098 → **0.0858** | −2.40 p.p. (CI linie [−2.55, −2.24]) | 0.40% |
| test/real v1.2, zamrożony test, top-5 + ocena Suryi | v4_200k_25k | 0.1040 → 0.0784 | −2.56 p.p. | 0.31% |

- Korektor daje ok. −2.5 do −2.8 p.p. CER (ok. 20% względnie) na v2 i v3.
- Na v4 (generator z ~50 fontami) zysk spada o połowę: −1.23 p.p. (−11%). 41% poprawek z v3 v4 robi sam,
  strojenie wybiera `L_min = 6`. Sam v4 (0.1098) jest lepszy niż v3 z korektorem (0.1200). Szczegóły: `results/v12/README.md`.
- Ocena kandydatów przez Suryę (teacher forcing, log P linii z kandydatem przy obrazie) przywraca zysk: −2.40 p.p. w LOGO.
  Surya dobrze wybiera między kandydatami, ale sama prawie nigdy nie woli poprawki od własnego odczytu. Lookahead w modelu
  jest wyłączony (`multi_output_distance = 0`), zadanie `ocr_with_boxes` nie zmienia CER.
- Lepszy model OCR i korektor częściowo naprawiają te same błędy: na v1 po korekcie v3 wygrywa z v2 tylko o 0.08 p.p.,
  na v1.1 o 0.43 p.p.
- Zysk jest na każdej z 8 grup, od −0.27 p.p. (grupa z najniższym CER) do −3.8 p.p.

## Co daje realny zysk (w kodzie, domyślnie włączone)

| składnik | efekt | dowód |
|---|---|---|
| słownik + kanał jednostajny (zwykły Levenshtein) | −0.70 p.p. | ablacja `--uniform` |
| wyuczone pomyłki 1:1 (model kanału) | dalsze −1.55 p.p. | `--uniform` vs `--no-multi` |
| jednostki wieloznakowe (`cz→w`, `ci→u`, `ls→b`) | dalsze −0.57 p.p., precision 0.785 → 0.835 | `--no-multi` |
| lista terminów anatomicznych TA (`en,la,pl`, tylko ochrona) | harm 0.71% → 0.31% przy tym samym CER | `results/RAPORT.md` §4b |
| `deloop` (higiena: zwijanie zapętleń i `<br>`) | v3 na RxHandBD 0.710 → 0.600 (−11.0 p.p.); notatki bez zmian | runda 2, niżej |

Wyuczone pomyłki to trzy czwarte zysku. Bez modelu kanału sam słownik daje mało.

## Co sprawdzono i nie dało zysku

| pomysł | wynik | decyzja |
|---|---|---|
| harm ≤ 0.5% zamiast 0.3% | +0.5 p.p. CER kosztem 2× harm | zostaje 0.3% |
| lista TA także przy strojeniu progów | CER −0.40 p.p., ale +114 błędnych podmian (`śluzowa→służbowe`) | odrzucone: płynny błąd gorszy od widocznego |
| większy słownik (sjp „odmiany”) | gorzej (−2.13 vs −2.29 p.p.), harm 0.62% | odrzucone |
| forward (suma ścieżek) zamiast Viterbi | w szumie, większy harm | Viterbi |
| opcja 2: tylko przywracanie diakrytyków | −0.09 p.p. (pułap oracle −0.36) | wchłonięta przez opcję 1 |
| klasy chronione P2–P8 | prawie bez wpływu (redundantne z regułą kwalifikacji) | zostają (tanie) |
| limit udziału jednej grupy w licznikach | 0.09 p.p., w szumie | zostaje (ochrona przed jednym zeszytem) |
| waga `w_diac` dla par diakrytycznych | w szumie (0.09862 vs 0.09874) | usunięta, prior jednostajny |
| kanał zależny od pozycji litery w słowie | v2 istotnie gorzej (+0.18…+0.35 p.p.), v3 nieistotnie lepiej (−0.10…−0.18) | odrzucone |
| osobne progi dla Tytułowych / WIELKICH | v3 −0.12 p.p., CI [−0.45, +0.12], progi niestabilne | odrzucone |
| ICD-11 jako słownik anatomiczny | pokrywa 12–32 z 273 słów GT spoza słownika | za mało; potrzebna odmiana przymiotników |
| listy PESEL (imiona, nazwiska) jako kandydaci | PERSON bez zmian; na notatkach chronią śmieci OCR (`mozna`, `sredni`) | odrzucone w tej formie |
| listy ICD jako kandydaci | w szumie | odrzucone |
| ochrona encji przez NER (HerBERT, nlp-ner) | F1 encji: GT 0.79 → OCR 0.69 → po korekcie 0.71; na notatkach 2.6× więcej fałszywych PERSON | odradzane jako ochrona |

## Obiecujące, ale niezweryfikowane na prawdziwych danych

- **Lista leków RPL jako kandydaci korekty** (z pseudo-licznością 5, bo próg `f_min=1` odcina słowa spoza NKJP):
  na syntetycznym tekście medycznym poprawność nazw leków 0.36 → 0.42, CER bez zmian. Szum syntetyczny pochodzi z tego
  samego modelu kanału co korektor, więc wynik jest optymistyczny. Kod nie wszedł do repo, czeka na test na
  prawdziwych receptach albo na wyrenderowanym tekście medycznym (generator pisma + Surya).
- **Większy limit szkód dla dokumentów medycznych:** na syntetyce przy harm 0.55% CER 0.082 i nazwiska 0.64,
  przy 1.27% CER 0.073 i nazwiska 0.71 (przy 0.3%: 0.090 i 0.59). Decyzja zależy od tego, co gorsze w dokumentacji:
  widoczny błąd OCR czy wiarygodne, błędne słowo.

## Obserwacje o danych i modelach (runda 2)

- **test/real v1.1 ma 68 linii z etykietą `[BLANK]`** (ramki bez transkrypcji, wszystkie na stronach `miesnie`).
  Z nimi CER rośnie do ~0.16. W ocenie wyłączone; do uzupełnienia albo usunięcia w zbiorze.
- Nowe strony v1.1 są trudniejsze: CER v3 na nich 0.137, na całym v1 0.120.
- **Surya v3 zapętla się poza dystrybucją** (RxHandBD, recepty z Bangladeszu): `Maring Maring …`, segmenty `<br>`.
  Na RxHandBD v3 ma CER 0.710, v2 0.596. Po `deloop` v3 ma 0.600.
- htr_lexicography (polskie karty słownikowe, core 2000 słów): CER bez rozróżniania wielkości liter 0.072 (v2 i v3).
  Etykiety upstream są małymi literami, model pisze wielkimi.
- Korektor na syntetyce: nazwiska 0.23 → 0.59, leki 0.30 → 0.36, choroby 0.27 → 0.48. Zepsucia to głównie
  skróty medyczne WIELKIMI (`TURP→TURĄ`, `FSH→FSM`), więc potrzebna jest lista skrótów medycznych.

## Zbiory zewnętrzne do testsetu v3 (przygotowane, nie wysłane)

- **RxHandBD** (Islam, CC BY 4.0 / MIT): 5578 wycinków słów, podział test 1115 / train 4463, 4 nieczytelne oznaczone,
  README z cytowaniem, licencją i listą zmian, manifest sha256.
- **htr_lexicography** (Idziak i in., brak licencji w repo): w zbiorze tylko skrypt pobierający, poprawki etykiet
  (mojibake → UTF-8, NFC), lista ID z sha256 i podzbiór core 2000; bez obrazów.

## Otwarte

1. Przełączenie punktu odniesienia na v3 na v1.1 (`data/baseline`, RAPORT, progi do wdrożenia).
2. Limit szkód dla dokumentów medycznych (wyżej).
3. Lista skrótów medycznych; odmiana przymiotników anatomicznych (`-owy`, przedrostki).
4. Render syntetycznego tekstu medycznego przez Suryę (serwer) i ponowny test listy RPL.
5. Logprobs Surya do oceny kandydatów (wymaga stacji 827).
