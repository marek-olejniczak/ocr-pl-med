# Narzędzie do wspomaganego labelowania HTR — dokumentacja techniczna

Dokumentacja prac nad rozszerzeniem narzędzia `labeling_tool.py` o automatyczną
pre-anotację (detekcja linii + rozpoznawanie pisma odręcznego).

**Kontekst:** ręcznie oznaczane notatki stanowią zbiór testowy dla docelowego
modelu inżynierki. Opisywany tu model jest wyłącznie **narzędziem pomocniczym**
przyspieszającym anotację — nie podlega ewaluacji jako model finalny.

---

## 0. Uruchomienie labelera z modelami zespołu

W repozytorium znajdują się wagi detektora YOLO
(`models/yolo_lines_best_iou_median.pt`) oraz adapter Surya LoRA
(`models/surya_lora_adapter/`). Poniższe polecenia należy wykonać w katalogu
repozytorium. Wymagany jest Python 3.10-3.13; dla workera używany jest 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python3.13 -m venv .venv-surya17
.venv-surya17/bin/python -m pip install surya-ocr==0.17.1 transformers==4.56.1 peft flask

HTR_DETECTOR=yolo HTR_BACKEND=surya_lora SURYA_DEVICE=cpu python labeling_tool.py
```

Interfejs: `http://127.0.0.1:5000`. Dla karty NVIDIA należy zamienić
`SURYA_DEVICE=cpu` na `SURYA_DEVICE=cuda`, a dla Apple Silicon na
`SURYA_DEVICE=mps`. Worker Surya startuje automatycznie na porcie 5055.
Pierwsze uruchomienie rozpoznawania pobiera bazowy model Surya z internetu;
dołączony adapter nie zastępuje modelu bazowego. Nie trzeba instalować
Ollamy ani pobierać TrOCR. Bez `HTR_BACKEND=surya_lora` domyślnie wybierany
jest lokalny TrOCR, którego wag ten zestaw nie zawiera.

Po wczytaniu PDF lub obrazu przycisk auto-anotacji uruchamia detekcję i OCR.
Propozycje można poprawiać, zatwierdzać i zapisywać w katalogu `dataset/`.
Istniejący zbiór można osobno pobrać przez `dvc pull` po skonfigurowaniu
własnego dostępu do magazynu DVC. Dane i środowiska nie są częścią tego commita.

Poniższe sekcje opisują także wcześniejsze eksperymenty z CRAFT i TrOCR;
aktualny zestaw do uruchomienia to YOLO + Surya LoRA z polecenia powyżej.

---

## 1. Architektura

Pipeline dzieli się na dwa niezależne, wymienne moduły:

```
strona → [DETEKCJA] → prostokąty linii → [ROZPOZNAWANIE] → tekst → [WERYFIKACJA]
```

| moduł | zadanie | metryka | rozwiązanie |
|---|---|---|---|
| Detekcja | znalezienie ramek linii | precision / recall / F1 | CRAFT (EasyOCR) |
| Rozpoznawanie | odczyt tekstu z ramki | CER / WER | TrOCR (dostrajany) |
| Weryfikacja | akceptacja lub korekta przez człowieka | — | interfejs webowy |

Wymienność backendów (`HTR_DETECTOR`, `HTR_BACKEND`) umożliwia porównywanie
metod bez zmian w pozostałej części systemu.

---

## 2. Charakterystyka zbioru

| cecha | wartość |
|---|---|
| oznaczonych linii | 2580 |
| stron | 155 |
| mediana długości tekstu | 15 znaków |
| maksymalna długość | 84 znaki |
| unikalnych znaków | 92 (pełny zestaw polskich diakrytyków) |

### 2.1. Istotne obserwacje dotyczące danych

**Dwa typy stron.** Zbiór zawiera odręczne notatki (pełne strony pisma) oraz
drukowane formularze medyczne z odręcznymi wpisami. W formularzach oznaczane są
**wyłącznie fragmenty odręczne** — np. na jednej stronie ankiety szpitalnej
oznaczono tylko dwa wpisy (`172`, `87`).

**Konsekwencja metodologiczna:** zadaniem jest detekcja **pisma odręcznego**, a
nie detekcja tekstu w ogóle. Uniwersalne detektory (CRAFT, Surya) wykrywają
również druk, co przy naiwnej ewaluacji generuje pozorne fałszywe alarmy.
Formularze wyłączono z benchmarku (`--exclude "KARTA PACJENTA"`).

**Trudność układu.** Notatki zawierają tekst rozproszony wokół rysunków
anatomicznych, linie nachodzące na siebie w pionie oraz papier w kratkę.

---

## 3. Detekcja linii

### 3.1. Porównanie metod

Warunki: 20 stron notatek odręcznych, ground truth z anotacji ręcznych.

| metoda | F1 (pokrycie) | precision | recall | czas/stronę |
|---|---|---|---|---|
| **CRAFT (nastrojony)** | **0.712** | 0.623 | 0.831 | 2.9 s |
| Surya | 0.681 | 0.578 | 0.829 | **0.61 s** |
| morfologia (heurystyka) | 0.029 | 0.012 | 0.064 | 0.2 s |

Wybrano **CRAFT** — najwyższe F1 przy braku dodatkowych zależności
(Surya wymaga Pythona ≥ 3.10, czyli osobnego środowiska).

### 3.2. Dlaczego heurystyka morfologiczna zawiodła

Pierwotna implementacja (binaryzacja → dylatacja pozioma → kontury) osiągnęła
F1 = 0.013. Diagnoza wykazała dwie przyczyny:

1. **Linie papieru w kratkę** scalały całą stronę w jeden obiekt. Dodano
   usuwanie długich ciągów poziomych i pionowych (`_remove_ruling_lines`).
2. Nawet po naprawie: składowe spójne to fragmenty liter (mediana 16 px) wobec
   linii o wysokości ~62 px, a linie **nachodzą na siebie pionowo**. Metody
   oparte na projekcji i morfologii nie radzą sobie z takim układem.

Wniosek: dla pisma odręcznego o nieregularnym układzie konieczny jest
detektor wyuczony.

### 3.3. Strojenie progów CRAFT

Progi domyślne biblioteki dobrano pod tekst drukowany. Wyniki strojenia
(12 stron, dopasowanie 1:1, IoU 0.3):

| text_threshold | low_text | link_threshold | F1 |
|---|---|---|---|
| 0.2 | 0.10 | 0.4 | 0.224 |
| 0.3 | 0.15 | 0.4 | 0.257 |
| 0.4 | 0.25 | 0.4 | 0.284 |
| 0.6 (domyślne) | 0.35 | 0.4 | 0.362 |
| 0.7 | 0.45 | 0.4 | 0.417 |
| **0.8** | **0.5** | **0.5** | **0.481** |
| 0.85 | 0.6 | 0.6 | 0.385 |
| 0.9 | 0.65 | 0.6 | 0.118 |

**Wynik kontrintuicyjny:** obniżanie progów pogarsza jakość — regiony detekcji
zlewają się w obiekty obejmujące wiele linii. Optimum leży **powyżej** wartości
domyślnych; przekroczenie 0.85 powoduje załamanie precyzji.

**Weryfikacja na rozłącznym zbiorze stron** (14 stron, inne ziarno losowania):

| konfiguracja | precision | recall | F1 |
|---|---|---|---|
| domyślna (0.6 / 0.35 / 0.4) | 0.238 | 0.462 | 0.314 |
| nastrojona (0.8 / 0.5 / 0.5) | 0.265 | 0.519 | 0.351 |

Strojenie generalizuje się (+12 % F1), choć zysk jest mniejszy niż na zbiorze
użytym do strojenia (0.481) — różnica ilustruje ryzyko przeuczenia
hiperparametrów.

### 3.4. Dobór metryki

Zastosowano dwie metryki, ponieważ dopasowanie 1:1 według IoU systematycznie
zaniża wynik przy różnicy granularności:

- **Dopasowanie 1:1 (IoU)** — klasyczne, zachłanne. Karze podwójnie sytuacje,
  gdy anotacja ręczna dzieli linię wizualną na segmenty semantyczne
  (np. termin polski i angielski w tej samej linii jako osobne ramki), a
  detektor zwraca jedną ramkę.
- **Pokrycie** — ramka GT uznana za trafioną, gdy detekcje pokrywają ≥ 70 % jej
  powierzchni; detekcja uznana za błędną, gdy ≥ 50 % jej powierzchni leży poza
  tekstem. Odporna na podziały i scalenia.

Różnica jest znacząca: CRAFT osiąga F1 0.479 (IoU) wobec 0.712 (pokrycie) na
tych samych danych. Do oceny przydatności praktycznej właściwsza jest metryka
pokrycia — anotatorowi wystarczy przybliżona ramka.

---

## 4. Rozpoznawanie tekstu

### 4.1. Odrzucenie modelu wizyjno-językowego

Testowano `qwen3-vl:8b` uruchamiany lokalnie przez Ollama.

**Wynik: nieprzydatny.** Model działa w trybie rozumowania — cały budżet
tokenów (`num_predict`) zużywa na wewnętrzne rozumowanie prowadzone po
angielsku, zwracając puste pole odpowiedzi (`done_reason=length`). Wyłączenie
trybu (`think: false`) nie działa dla tego modelu w Ollama.

Istotne: w polu `thinking` widoczny był **poprawny odczyt** — model odczytuje
pismo prawidłowo, lecz nie sposób wydobyć wyniku. Dodatkowo czas ~11 s na linię
(~4 min na stronę) dyskwalifikuje go w zastosowaniu interaktywnym.

### 4.2. Dostrajanie TrOCR

Wybrano TrOCR-base (334 M parametrów) — architektura seq2seq dedykowana
zadaniu „obraz linii → tekst", natywna dla biblioteki `transformers`.

Uzasadnienie wyboru modelu małego zamiast dużego VLM:
- zadanie jest wąskie (jeden styl pisma, jedna domena),
- czas ~0.5 s na linię wobec 11 s (ok. 20× szybciej),
- ogólna wiedza dużego modelu jest w tym zastosowaniu nieużyteczna.

**Jakość zero-shot** (25 linii, przed dostrajaniem):

| model bazowy | CER |
|---|---|
| `microsoft/trocr-base-handwritten` | 0.5756 |
| `PET3R12/trocr-base-polish-handwriting` | 0.6111 |

Model polski wypada nieznacznie gorzej pod względem CER bez dostrajania, lecz
jakościowo wykazuje lepsze dopasowanie do języka (`dolny` → `dolny` wobec
`doing`).

### 4.3. Porównanie modeli bazowych po dostrojeniu

Przeprowadzono dwa treningi na tym samym zbiorze (2580 linii):

| | baza angielska | baza polska |
|---|---|---|
| CER po 1. epoce | 0.7721 | **0.1305** |
| **najlepszy CER** | 0.1118 | **0.0470** |
| epoka najlepszego wyniku | 8 | 10 |
| stabilność | skok straty w 10. epoce | bez zakłóceń |

**Model pretrenowany na języku polskim okazał się wyraźnie lepszy** — CER
niższy 2.4-krotnie. Różnica jest widoczna już po pierwszej epoce, co wskazuje na
przenoszenie reprezentacji polskich znaków diakrytycznych. Dominującym błędem
wariantu angielskiego było gubienie znaku `Ó`.

Konfiguracja drugiego treningu obejmowała dodatkowo: wygaszanie kosinusowe
współczynnika uczenia z rozgrzewką, przycinanie gradientu, wczesne zatrzymanie
oraz augmentację kadrowania (opisaną w 4.4). Zmiana kilku czynników naraz
oznacza, że nie można przypisać całej poprawy wyłącznie modelowi bazowemu —
rozdzielenie wpływów wymagałoby dodatkowego treningu bez augmentacji.

### 4.4. Przesunięcie rozkładu między treningiem a użyciem

Podczas testów w interfejsie zaobserwowano jakość wyraźnie gorszą niż na
zbiorze walidacyjnym. Przyczyną było **przesunięcie rozkładu**: model trenowano
na wycinkach z ramek rysowanych ręcznie (ciasnych, dobrze wykadrowanych), a
w działającym narzędziu otrzymywał wycinki z ramek detektora — kadrowane
inaczej, czasem ucinające znaki lub obejmujące sąsiedni tekst.

Zastosowano **augmentację kadrowania**: linie wycinane są z oryginalnej strony
z losowym przesunięciem krawędzi (±3 % szerokości, ±12 % wysokości), co
upodabnia dane treningowe do wyjścia detektora.

**Pomiar wpływu dopasowania ramki na CER** (model polski, ramki z detektora
CRAFT dopasowywane do ramek referencyjnych):

| strona | IoU > 0.3 | IoU > 0.5 | IoU > 0.7 |
|---|---|---|---|
| Noga_page04 | 0.3197 | 0.0929 | **0.0147** |
| Noga_page13 | 0.3288 | **0.0127** | 0.0180 |
| Noga_page05 | 0.5494 | 0.2727 | — |

Wniosek: przy dobrze dopasowanej ramce model osiąga CER rzędu 1.5 %. Obserwowany
w praktyce błąd rzędu 20-30 % wynika **niemal wyłącznie z niedopasowania ramek**,
a nie z jakości rozpoznawania. Kierunek dalszej optymalizacji prowadzi zatem
przez poprawę detekcji i możliwość korekty ramki przez użytkownika, nie przez
dalsze dostrajanie modelu rozpoznającego.

---

## 5. Rejestrowanie pochodzenia anotacji

Plik `annotations.csv` rozszerzono o kolumnę `source`:

| wartość | znaczenie |
|---|---|
| `manual` | ramka narysowana i opisana ręcznie |
| `auto` | propozycja modelu, niezweryfikowana |
| `verified` | propozycja zaakceptowana bez zmian |
| `corrected` | propozycja poprawiona przez człowieka |

Umożliwia to obliczenie **wskaźnika korekt** (odsetek `corrected` wobec
`verified`) — miary użyteczności narzędzia mierzącej cały pipeline, a nie
pojedynczy komponent. Jest to metryka właściwsza do oceny wsparcia procesu
anotacji niż F1 detektora.

---

## 6. Środowisko i napotkane problemy techniczne

### 6.1. Konfiguracja

| środowisko | Python | przeznaczenie |
|---|---|---|
| `.venv` | 3.9 | narzędzie (Flask, CRAFT, PyMuPDF) |
| `.venv-surya` | 3.13 | Surya, dostrajanie TrOCR (torch + MPS) |

Sprzęt: Apple M4 Pro, 24 GB pamięci zunifikowanej, akceleracja MPS.

### 6.2. Rozwiązane problemy

**Pillow a Python 3.14.** Surya wymaga `pillow<11`, dla którego brak
prekompilowanych pakietów dla Pythona 3.14. Rozwiązanie: Python 3.13.

**Tokenizer TrOCR w transformers 5.x.** Repozytoria TrOCR zawierają tokenizer w
formacie `vocab.json` + `merges.txt`, którego `transformers` 5.x nie konwertuje
automatycznie (błąd `Couldn't instantiate the backend tokenizer`). Rozwiązanie:
wczytanie tokenizera z `roberta-base` — TrOCR używa identycznego słownika, więc
identyfikatory tokenów pozostają zgodne (funkcja `load_processor`).

**Opóźnienie modelu a interaktywność.** Dodano tryb wsadowy przetwarzający
wszystkie strony w tle, dzięki czemu oczekiwanie na model nakłada się na czas
weryfikacji poprzednich stron.

---

## 7. Narzędzia pomocnicze

| plik | przeznaczenie |
|---|---|
| `auto_annotate.py` | detekcja linii i rozpoznawanie tekstu (wymienne backendy) |
| `bench_detectors.py` | benchmark detektorów wobec anotacji ręcznych |
| `export_training_data.py` | przygotowanie danych treningowych (manifest JSONL) |
| `train_trocr.py` | dostrajanie TrOCR z monitorowaniem CER |

Przykłady użycia:

```bash
# porównanie detektorów
python bench_detectors.py --detectors craft surya morph --pages 20 --only Noga

# przygotowanie danych i dostrajanie
python export_training_data.py
python train_trocr.py --epochs 12 --batch 8 --out trocr_finetuned_en
```

---

## 8. Wnioski

1. **Heurystyki obrazowe są niewystarczające** dla pisma odręcznego o
   nieregularnym układzie — różnica wobec metod wyuczonych wynosi ponad rząd
   wielkości (F1 0.03 wobec 0.71).

2. **Progi domyślne bibliotek wymagają strojenia** pod konkretną domenę,
   a kierunek zmian bywa kontrintuicyjny (podniesienie progów poprawiło recall).

3. **Dobór metryki wpływa na wnioski** — ta sama metoda uzyskuje F1 0.48 lub
   0.71 zależnie od sposobu dopasowania ramek. Metrykę należy dobierać do
   zastosowania, a nie odwrotnie.

4. **Duży model ogólny przegrywa z małym wyspecjalizowanym** w zadaniu wąskim:
   VLM o 8 mld parametrów okazał się nieprzydatny, podczas gdy model 334 M po
   dwóch epokach dostrajania osiągnął CER 0.17.

5. **Rejestrowanie pochodzenia anotacji** pozwala mierzyć rzeczywistą
   użyteczność narzędzia w procesie anotacji, niezależnie od metryk
   komponentów.

---

## 9. Kierunki dalszych prac

- porównanie dostrojonych wariantów: baza angielska wobec polskiej,
- pomiar wskaźnika korekt na rzeczywistej sesji anotacji,
- rozważenie architektur CTC (Kraken, PyLaia) — efektywniejszych przy
  ograniczonej liczbie przykładów treningowych,
- ewentualne włączenie zewnętrznego detektora ramek (do porównania w benchmarku).
