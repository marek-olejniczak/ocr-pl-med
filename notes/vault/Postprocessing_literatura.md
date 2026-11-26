# Postprocessing OCR — przegląd literatury

> **To jest research sprzed implementacji** (2026-10-29; sprostowania 2026-11-26). Co faktycznie zbudowaliśmy
> i zmierzyliśmy: [[Postprocessing_eksperymenty]]. Opcje z końca tego pliku to ówczesne plany, nie stan prac.
> Pomiar kandydatów z sekcji „Krok 1” (przeciek po zeszytach, tylko błędne słowa) **nie nadaje się do cytowania**
> jako wynik; zastępują go eksperymenty E1–E5.

Research z 2026-10-29. Cel: co można dołożyć za modelem OCR (Surya LoRA) w potoku aplikacji
dla polskich rękopisów medycznych (nazwy leków, słownictwo anatomiczne).

> Zastrzeżenia: nie znaleźliśmy prac o polskich rękopisach medycznych z LLM, więc rekomendacje dla rękopisów
> medycznych są naszym wnioskiem, nie wynikiem z literatury. Liczb z benchmarku HTR (2503.15195) nie czytałem
> z PDF-a, tylko zdanie z abstraktu. Link do oficjalnego pliku XML rejestru leków jest niepotwierdzony.

## Polski post-OCR: PolEval 2021, Task 3 (sprostowanie)

„Post-correction of OCR results”, [proceedings](https://poleval.pl/files/poleval2021.pdf),
[dane](https://github.com/poleval/2021-ocr-correction). **Druk**, nie rękopis: polskie książki z lat 1791–1998
(979 książek, ok. 69 tys. stron) rozpoznane Tesseractem; metryka WER (bez CER). Zwycięskie podejścia oparte na
mT5/plT5 zeszły z WER ok. 18 do ok. 3.7; heurystyki (Marcińczuk) do 8.2. Surowy WER różni się między artykułami
(16.5–18.7), więc liczby z różnych artykułów nie są wprost porównywalne. Licencja danych niejasna (niezweryfikowana).
Na nasz przypadek się nie przenosi (inny silnik, druk, inny rodzaj błędów); w pracy jako kontekst.

## Metody

| Metoda | Idea | Ocena dla nas |
|---|---|---|
| Kandydaci z macierzy pomyłek + słownik + ranking | Słowo rozszerzane przez macierz pomyłek, filtr słownikiem, ranking cechami | Wprost nasz pomysł; dane do macierzy już mamy |
| Model kanału zaszumionego | P(słowo) × P(błędy \| słowo), Viterbi | Klasyczna, tania baza; macierz pomyłek = P(błędy) |
| CTC + leksykon + KenLM | Ograniczenie dekodowania w trakcie | Tylko modele CTC, Surya jest autoregresyjna; zyski często małe |
| Dopasowanie do katalogu | Nazwy leków poprawiane wyszukiwaniem w katalogu | Najbliższy nasz przypadek użycia |
| Znakowy n-gram LM na wzorcach syntetycznych | Wzorce linii z recept nie występują w zwykłym tekście | Podobne do tego, co robi nasz generator |
| Fine-tuned ByT5 (seq2seq na bajtach) | Uczony na parach (zaszumione, poprawne) | Działa, ale musi pasować do silnika OCR |
| LLM zero-shot | Prompt „popraw” | Słabo |

## Co mówią źródła

- **Mapa podobieństw jako ograniczenie kandydatów** to standard: macierz pomyłek rozszerza słowo,
  słownik filtruje, ranker wybiera (Kissos & Dershowitz, arXiv 1604.06225). W chińskim spelling
  correction odpowiednikiem jest confusion set (kształt/fonetyka), z tą samą wadą: generowanie
  tylko z zestawu blokuje wszystko spoza niego (arXiv 2301.06323).
- **Model kanału zaszumionego** z n-gramami liter, prawdopodobieństwami pomyłek i bigramami słów
  przeszukiwany Viterbim: ACL W96-0108.
- **Dekodowanie z leksykonem/LM bywa mało warte**: w MXNet/Gluon CER 18.94 (greedy) → 18.90
  (leksykon) → 18.84 (beam + LM). W pracy o dekompozycji n-gramowej (arXiv 2012.14459) 4-gramowy
  LM znakowy dał CER 5.64 vs 5.60 greedy (gorzej), ale WER 18.14 vs 19.10 (lepiej). W Digital
  Peter (arXiv 2103.09354) duży n-gramowy LM: CER 3.7 → 3.5. Wniosek: poprawia raczej WER niż CER,
  zależnie od zbioru.
- **LLM zero-shot nie poprawia HTR**: „LLMs demonstrate limited ability to autonomously correct
  errors in zero-shot transcriptions” (arXiv 2503.15195). Samokorekta działa w OCR drukowanym
  lepiej niż w HTR.
- **Fine-tuned ByT5 działa**: pobił state-of-the-art w pracy z Twente (kontekst 50 znaków najlepszy;
  few-shot bez fine-tuningu nie działał) i poprawił CER/WER na zaszumionych odręcznych tekstach
  portugalskich (NeurIPS 2024 workshop, De Araújo).
- **Korektor musi pasować do silnika**: ByT5 dla Devanagari (arXiv 2606.29213) poprawiał błędy
  silnika, na którym go uczono, a na innych był neutralny lub szkodliwy. Dla nas: uczyć na błędach
  Surya i przeuczać po zmianie modelu OCR.
- **Recepty / nazwy leków**: NAACL 2025 industry (2025.naacl-industry.63) — literówki w nazwach
  leków to ryzyko dla pacjenta, poprawa przez retrieval z katalogu. Weakly supervised IE
  (arXiv 2306.06823) — znakowy n-gram LM budowany z syntetycznych wzorców linii.
- **Polskie LLM-y**: Bielik v3 Small ma warianty 1.5B i 4.5B (arXiv 2505.02550). PLLuM według tego
  raportu zaczyna od 8B. Brak wyników dla korekty OCR.
- **Klasyk**: Broda & Piasecki 2007 (IDEAL) — korekta OCR pisma medycznego przez podobieństwo
  semantyczne na polskich dokumentach medycznych; przed erą LLM, nie czytałem szczegółów.
- **Przegląd**: Survey of Post-OCR Processing Approaches (ACM CSUR 2021, DOI 10.1145/3453476).

## Ryzyko

Przy nazwach leków zamiana poprawnego, rzadkiego słowa na popularne jest groźniejsza niż
zostawienie błędu OCR. LLM nadaje się co najwyżej jako ranker kandydatów z zamkniętej listy,
nie jako generator.

## Opcje dla potoku (od najtańszej)

1. Słownik + model kanału z naszą macierzą pomyłek i rankingiem.
2. Osobne traktowanie diakrytyków (dalszy eksperyment).
3. Korekta tylko tokenów o niskiej pewności modelu OCR (wymaga log-probów z Surya).
4. Fine-tuning ByT5-small na syntetycznych parach z błędami wstrzykniętymi z naszej macierzy.
5. Bielik 1.5B/4.5B jako ranker kandydatów z punktu 1.

## Co z analizy błędów (branch `error-analysis`, Surya v2_200k, handlabeled 2160 linii)

CER 0.128, WER 0.48. Zbiór in-sample względem wyboru checkpointu.

- Bilans: 4014 substytucji, 914 usunięć, 474 wstawień (Levenshtein).
- Skoncentrowane: `k→h` 448, `a→o` 244, `ś→s` 108, `t→ł` 102.
- Diakrytyki: `ś→s` w 82% błędów znaku `ś`, `ż→z` 61%, `ó→o` 56%, `ę→e` 42%, `ą→a` 39%.
- Wieloznakowe (609 zdarzeń po ważonym wyrównaniu): `ci→u` 94, `cz→w` 87, `rz→n` 65, `ci→w` 47.
  Zwykła macierz 1:1 ich nie pokrywa.
- Część pomyłek zależy od autora/zeszytu (np. `cz→w` w 51% z jednego zrzutu) — model błędu może
  się słabo przenosić na nowe pismo.
- Gotowe dane: `training/ocr/error_analysis/results/chars/confusion_counts.csv`,
  `results/ngram/multi_events.csv`, `confusion_counts_clean.csv`.

## Źródła

- https://zenodo.org/records/4635569
- https://arxiv.org/pdf/1604.06225
- https://preview.aclanthology.org/moar-dois/W96-0108.pdf
- https://arxiv.org/abs/2503.15195
- https://essay.utwente.nl/essays/102117
- https://neurips.cc/virtual/2024/107779
- https://arxiv.org/pdf/2606.29213
- https://preview.aclanthology.org/moar-dois/2025.naacl-industry.63.pdf
- https://arxiv.org/pdf/2306.06823
- https://arxiv.org/pdf/2012.14459
- https://arxiv.org/pdf/2103.09354
- https://arxiv.org/pdf/2301.06323
- https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67
- https://arxiv.org/html/2505.02550v2

## Krok 1: słownik i mapa podobieństw (research 2026-10-29)

### Słowniki

| Zasób | Zawartość | Licencja / dostęp | Uwagi |
|---|---|---|---|
| PoliMorf (SGJP + Morfologik) | ~3.8 mln form (wersja 0.6.7: 6.58 mln wierszy form/tagów), pełna fleksja | BSD 2-clause; https://zil.ipipan.waw.pl/PoliMorf (0.6.7), CLARIN-PL: polimorf-20180722.tab.gz | Format `forma  lemat  tag  kategoria`, bez częstości. Morfologik oznacza część danych heurystycznie (niska wiarygodność niektórych cech gramatycznych, nie form). |
| NKJP unigramy | 5.36 mln wpisów `częstość słowo`, małe litery, z szumem | CC-BY; https://zil.ipipan.waw.pl/NKJPNGrams (`1grams.gz`) | Źródło częstości P(w). Filtrować do liter polskich i progu częstości. |
| Rejestr Produktów Leczniczych | 22 724 wiersze, 11 463 nazwy, 10 454 substancje | `nlp-ner/data/raw/Rejestr_Produktow_Leczniczych_calosciowy_stan_na_dzien_20260527.csv` (separator `;`) | Mamy lokalnie; oficjalny URL niepotwierdzony. |
| Listy z `nlp-ner/data` | choroby (16 387), badania (10 001), leki (11 464), zabiegi | lokalnie | Zawierają wielowyrazowe nazwy, tokenizować na słowa. |
| ICD-10 PL | klasyfikacja chorób | CeZ / Rejestr Systemów Kodowania, XML na dane.gov.pl (stan 2021, CSV niepotwierdzony) | Do rozważenia, pokrywa się z `diseases.csv`. |
| Polski MeSH (MSHPOL) | ~31 tys. haseł | UMLS, wymaga licencji | Praca z 2014: 70% terminów z tekstów klinicznych nie ma w polskim MeSH. Nie do użycia wprost. |
| Hunspell pl_PL, Terminologia Anatomica PL | | nie sprawdzone | Do sprawdzenia, jeśli PoliMorf nie wystarczy. |

Inne źródła częstości (nie sprawdzane): Leipzig Corpora Collection (CC BY 4.0), SUBTLEX-PL, listy z OpenSubtitles (CC BY-SA 4.0).

### Mapa podobieństw wizualnych

- **Gotowe mapy ogólne** istnieją, ale nie dla rękopisu: macierz Montoya/Simpson dla liter łacińskich
  (oceny ludzi, pismo drukowane), Unicode confusables (reguły, dla drukowanych glifów), embeddingi
  ViT (EMNLP 2023, "Quantifying Character Similarity with Vision Transformers") z cosinusem jako
  podobieństwem, normalizacja podobieństw do prawdopodobieństw (patent US 12592965).
  Synteza błędów OCR z próbkowaniem proporcjonalnym do podobieństwa glifów: arXiv 2408.02253.
- **Dla rękopisu najlepsza jest mapa z błędów własnego modelu**: macierz pomyłek uczona na wyjściu
  silnika (papier ICADL 2018; Park & Govindaraju, CVPR 2000 — podobieństwo leksykalne uwzględniające
  separowalność klas; patenty US 5802205, US 5963666 — macierz pomyłek jako model błędu i
  macierz zamian normalizowana kolumnami). Literatura dla k/h w rękopisie łacińskim: brak.
- **Wniosek**: nasza macierz z `error-analysis` jest tą właściwą mapą. Mapę ogólną (ViT/font)
  traktować jako prior wygładzający dla par, których nie widzieliśmy, nie jako zamiennik.

### Pomiar na predykcjach Surya v2_200k (handlabeled, 2160 linii)

Predykcje: `/home/marek/put/inzynierka/heatmap/data/predictions.jsonl` (2026-10-01, ten sam adapter co raport w `error-analysis`).

- Pokrycie słownikiem tokenów GT (5732 tokenów, 1701 unikalnych): PoliMorf 92.0%, +RPL/ner/NKJP>=100 93.5%.
  Same listy medyczne: RPL 15.7%, ner 57.4%. Handlabeled to głównie notatki anatomiczne, nie recepty,
  więc listy leków prawie nic tu nie wnoszą (to nie znaczy, że nie wniosą na receptach).
- Poza słownikiem: angielskie/łacińskie terminy (muscle, femur, surface), słowa bez diakrytyków
  w GT (kosci, posladkowa) i skróty (zewn, przyśr). Trzeba sprawdzić, czy GT bez diakrytyków to
  błąd etykiet czy cecha zeszytu.
- 10.8% błędnych tokenów to poprawne słowa słownika (real-word errors). Słownik ich nie wykryje.
- Odległość Levenshteina (GT, pred) błędnych tokenów: 1: 47%, 2: 29%, 3: 14%, >=4: 9%.
- Eksperyment (400 błędnych tokenów z GT w słowniku, macierz z parzystych linii, test na nieparzystych;
  przecieka po zeszytach, więc wynik optymistyczny):

| | zwykły Levenshtein | macierz pomyłek |
|---|---|---|
| kandydaci, średnio (Lev<=2) | 109 (mediana 35) | 49 (mediana 2) przy koszcie <= koszt(GT)+2 nat |
| kandydaci, średnio (Lev<=3) | 1472 (mediana 638) | 97 (mediana 7) przy koszcie <= koszt(GT)+4 nat |
| odzyskanie GT: Lev<=1 / 2 / 3 | 43% / 74% / 89% | |
| top-1 / top-5 (+ częstość NKJP) | 38% / 62% | 57% / 74% |

Tego eksperymentu nie mierzy się dla poprawnych tokenów, więc nie wiemy jeszcze, ile dobrych słów
by zepsuł. To najważniejszy brakujący pomiar.
