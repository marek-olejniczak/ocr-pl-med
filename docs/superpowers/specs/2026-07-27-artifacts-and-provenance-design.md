# Artefakty na dokumencie + historia zbiorów danych

Data: 2026-07-27
Status: zatwierdzony przez Tomka

## Cel

Trzy nowe augmentacje symulujące realne ślady na papierowym formularzu
(przypadkowe kreski długopisem, pieczątki, zakreślanie markerem), sanityzacja
treści pod kątem pokrycia znaków przez czcionki, oraz system historii
wygenerowanych zbiorów danych.

Kontekst: generator produkuje dane dla dwóch modeli — detekcji linii (YOLO)
i OCR. Ground truth zawiera bbox + transkrypcję każdej linii. Do puli
czcionek doszło 9 własnych fontów zbudowanych z prawdziwego pisma
(Marek 1–8, Pokladowski), łącznie 15 użytecznych.

## 1. Sanityzacja treści (warunek poprawności danych)

**Problem:** nowe czcionki nie zawierają kilku znaków, które produkuje nasze
słownictwo. Renderują się jako pusty prostokąt (.notdef), podczas gdy
transkrypcja w ground truth twierdzi, że znak tam jest — trucizna dla OCR.
Zmierzona częstość na 15 000 wygenerowanych pól: `†` 0.96%, `[`/`]` 0.30%,
`µ` 0.12%, `–`/`‑` 0.06%. Razem ~1.3% pól.

**Rozwiązanie — nowy moduł `src/text_sanitize.py`, dwa etapy:**

1. `normalize_for_handwriting(text) -> str` — zamiana na to, co człowiek
   faktycznie napisałby ręcznie:

   | Wejście | Wyjście |
   |---|---|
   | `‑` (U+2011), `–` (U+2013), `—` (U+2014) | `-` |
   | `µ` (U+00B5), `μ` (U+03BC) | `u` |
   | `†` (U+2020), `‡` (U+2021) | usunięte |
   | `[` `]` | `(` `)` |
   | `„` `”` `“` (U+201E/201D/201C) | `"` |
   | `’` (U+2019) | `'` |
   | ` ` (U+00A0, nbsp) | zwykła spacja |

   Po zamianach: sklejenie powstałych podwójnych spacji i przycięcie brzegów.

2. `strip_unsupported(text, font_path) -> str` — siatka bezpieczeństwa:
   usuwa każdy znak, którego dany font nie ma w tablicy `cmap`. Tablice
   czytane raz na font przez `fontTools` i cache'owane w słowniku
   modułowym. Dzięki temu problem nie wróci przy dorzucaniu kolejnych
   czcionek.

**Punkt zastosowania:** `generate_field_content` dostaje nowy parametr
`font_path` i sanityzuje treść **przed** pomiarem szerokości. Kluczowe:
transkrypcja zapisana w ground truth to dokładnie ten sam string, który
poszedł do renderu. Ta sama sanityzacja obowiązuje treść pieczątek.

## 2. Kolejność w pipeline

```
wypełnienie tekstem  →  pieczątka  →  marker  →  kreski  →  symulacja skanu
     (records)          (+records)    (bez GT)   (bez GT)     (bez zmian GT)
```

Wszystkie trzy artefakty to fizyczne ślady na papierze, więc powstają
**przed** symulacją skanu — profil szarości/ksero działa na nich tak samo
jak na reszcie kartki. Żaden nie zmienia geometrii obrazu, więc istniejące
bboxy pozostają ważne.

Kolejność wewnętrzna wynika z rzeczywistości: pieczątkę przybija się na
gotowym dokumencie, marker nakłada się na istniejący tekst, a przypadkowa
kreska długopisem może przejechać po wszystkim.

Implementacja: nowy moduł `src/artifacts.py` z trzema publicznymi funkcjami
(`draw_stamp`, `draw_highlights`, `draw_stray_strokes`). `fill_form.py` ma
już ~700 linii i nie powinien rosnąć dalej.

## 3. Kreski długopisem

Prawdopodobieństwo **20%** na dokument; przy trafieniu 1–3 kreski.

- Kolor: ten sam atrament co pismo na formularzu (ta sama osoba, ten sam długopis)
- Grubość 1–3 px, rysowane jako gładka łamana z dryfującym kątem
- Trzy rodzaje losowane równomiernie:
  - **długie pociągnięcie** — 150–500 px, lekko wygięte
  - **krótkie muśnięcie** — 30–120 px
  - **zawijas** — ciasny zygzak/bazgroł o szerokości 40–150 px
- Mogą przecinać istniejący tekst i wychodzić poza pola formularza
- **Zero bboxów** — czyste negatywne przykłady; model uczy się, że nie każdy
  ciemny ślad na kartce jest linią tekstu

Metadane: `{"count": n, "kinds": [...]}`.

## 4. Pieczątki

Prawdopodobieństwo **30%** na dokument, maksymalnie jedna pieczątka.
Kształt: **70% prostokątna**, **30% okrągła**. Kąt obrotu: **±20°**.

**Treść** (z istniejącego `Vocabulary`, po sanityzacji):
nazwa poradni/oddziału, `lek. med. <imię nazwisko>`, `spec. <specjalizacja>`,
`nr prawa wyk. zawodu <7 cyfr>`, opcjonalnie NIP/REGON.

**Wygląd:**
- Czcionka drukowana: DejaVu Sans (Regular/Bold), dodana do repo w
  `resources/fonts_print/` — licencja pozwala na redystrybucję, a repo ma
  pozostać samowystarczalne (bez zależności od matplotlib)
- Rozmiar tekstu 14–22 px
- Prostokątna: ramka 1–2 px (czasem podwójna), 2–4 linie tekstu,
  220–420 px szerokości, 70–160 px wysokości
- Okrągła: okrąg zewnętrzny + wewnętrzny, tekst po łuku u góry, tekst
  prosty w środku, średnica 180–300 px
- Kolor tuszu: niebieski, czarny lub czerwony
- **Niedobicie:** maska gładkiego szumu sprawia, że część farby nie odbiła
  się równo (typowe dla pieczątek); dodatkowo 15% szans na przycięcie
  fragmentu przy krawędzi oraz lekki blur 0.3–0.6

**Umiejscowienie:** preferencyjnie dolna 1/3 strony (tam, gdzie na
prawdziwych skierowaniach stoi pieczątka przy podpisie). Do 8 prób
losowania pozycji; próba odrzucana, jeśli pieczątka zasłoniłaby więcej niż
15% powierzchni któregokolwiek istniejącego bboxa. Gdy żadna próba się nie
powiedzie — pieczątka pomijana.

**Ground truth:** bboxy tylko dla pieczątek **prostokątnych** o kącie
**|α| ≤ 5°**. Wtedy każda linia tekstu dostaje własny bbox i transkrypcję
z `source: "stamp"`. Bbox liczony przez istniejące `rotate_bbox` (obrót
czterech rogów tightowego bboxa linii, obwiednia osiowa) — przy 5° narzut
na wysokość to ~+10%. Pieczątki okrągłe oraz prostokątne o |α| > 5° są
czystym szumem bez bboxów.

Metadane: `{"shape": ..., "angle_deg": ..., "color": [...], "in_gt": bool,
"lines": n}`.

## 5. Marker (zakreślacz)

Prawdopodobieństwo **12%** na dokument; przy trafieniu 1–3 zakreślenia.

- **Cele:** istniejące rekordy o `source` ∈ {`synthetic`, `printed`} —
  zarówno wpisane pismo, jak i drukowane etykiety formularza
- Kolory: żółty, zielony, różowy, pomarańczowy
- Tryb mieszania: **multiply** — ciemny atrament zostaje ciemny, biały
  papier przyjmuje kolor
- Realizm: pasmo wyższe niż tekst (rozszerzenie 2–6 px w pionie),
  przestrzelenie początku/końca o −8…+12 px, faliste krawędzie
  (sinusoida + szum), czasem 1–2° przekrzywienia
- **Twardy limit czytelności:** przy mnożeniu koloru `C` przez biały papier
  wynikiem jest samo `C`, więc reguła brzmi: luminancja koloru markera
  (`0.299R + 0.587G + 0.114B`) musi wynosić **≥ 150**. Kolor o niższej
  luminancji jest rozjaśniany w kierunku bieli aż do spełnienia warunku.
  Dzięki temu po `photocopy_contrast` (który ścina wartości poniżej 80–115
  do czerni) pasek pozostaje szary, a nie czarny, i pismo pod spodem nadal
  się czyta — etykieta OCR zostaje prawdziwa
- **Ground truth bez zmian** — tekst pod paskiem nadal tam jest, bboxy
  zostają nietknięte

Metadane: `{"count": n, "colors": [...], "targets": [indeksy rekordów]}`.

## 6. Historia zbiorów danych

### `dataset_card.json` — karta w folderze każdego zbioru

Zapisywana automatycznie na koniec generacji przez nowy moduł
`src/dataset_card.py`:

```json
{
  "name": "12k_records",
  "created": "2026-07-27T14:32:11",
  "git_commit": "2b208f2",
  "git_dirty": false,
  "command": "python src/generate_yolo_dataset.py --templates-dir templates ...",
  "seed": 2026,
  "counts": {"images": 12000, "annotations": 664860, "templates": 40},
  "sources": {"printed": 498300, "synthetic": 143267,
              "handwritten": 23293, "stamp": 4127},
  "fonts": ["Caveat-Regular.ttf", "Marek_1-Regular.ttf", "..."],
  "augmentations": {
    "stamp_prob": 0.30, "stroke_prob": 0.20, "highlight_prob": 0.12,
    "pen_fade_prob": 0.15, "empty_field_range": [0.0, 0.40],
    "form_font_size_range": [26, 40], "v_overflow_frac": 0.18,
    "scan_profiles": [["clean_color", 0.45], ["grayscale", 0.35],
                      ["photocopy", 0.20]]
  },
  "observed": {
    "scan_profiles": {"clean_color": 5405, "grayscale": 4286,
                      "photocopy": 2309},
    "bases": {"blank": 9011, "partial": 2989},
    "multiline_fields": 2486, "stamps": 3612,
    "strokes": 2401, "highlights": 1455
  },
  "note": "opcjonalny tekst z --note"
}
```

Sekcja `augmentations` jest zrzucana z jednej funkcji
`augmentation_settings()` czytającej stałe wprost z kodu (prawdopodobieństwa,
zakresy, profile skanu, rozmiary pisma). Nie ma ręcznego przepisywania, więc
karta nie może się rozjechać z rzeczywistym zachowaniem generatora.

`git_dirty` mówi, czy w momencie generacji drzewo robocze miało
niezacommitowane zmiany — bez tego `git_commit` bywa mylący.

### `docs/datasets.md` — rejestr w repo

Tabelka wersjonowana w gicie, jeden wiersz na zbiór: data, nazwa, commit,
seed, liczba obrazów, liczba anotacji, notatka. Dopisywana automatycznie po
każdej generacji; ponowna generacja pod tą samą nazwą **nadpisuje wiersz**
zamiast dublować (klucz: nazwa zbioru).

### CLI

Dwa nowe argumenty `generate_yolo_dataset.py`:
- `--dataset-name` (domyślnie: nazwa katalogu wyjściowego)
- `--note` (opcjonalny opis, ląduje w karcie i w rejestrze)

### Zbiór istniejący

Dla wygenerowanego wcześniej `output/12k_records` powstaje karta
retrospektywnie, na podstawie znanych parametrów (seed 2026, commit
`2b208f2`, komenda z historii) i statystyk policzonych z jego
`annotations.json` oraz `metadata/`. W rejestrze dostaje notatkę, że
powstał przed wprowadzeniem nowych augmentacji.

## 7. Próbka do przeglądu

Po zaimplementowaniu sanityzacji i trzech augmentacji (a **przed** pracą nad
historią zbiorów) powstaje mała próbka ~20 obrazów z wizualizacją bboxów,
wymuszająca wystąpienie wszystkich nowych efektów, na nowych czcionkach.
Tomek ją ogląda i akceptuje albo zgłasza poprawki, zanim praca pójdzie
dalej. To bramka przeglądowa, nie formalność.

## Poza zakresem

- Rotacja strony — pozostaje trwale wyłączona
- Zaginięcia kartki, plamy po kawie, tekstura papieru (kod istnieje,
  celowo nieużywany — formularze mają własne tło)
- Skreślenia i poprawki tekstu (osobny pomysł, na później)
- Odchudzanie słownika ICD-10 (sanityzacja rozwiązuje problem znaków)
- Ponowna generacja dużego zbioru — decyzja Tomka po obejrzeniu próbki

## Kryteria sukcesu

1. Żaden wygenerowany obraz nie zawiera znaku `.notdef` (pustego
   prostokąta): test zbiera zestaw znaków produkowanych przez słownictwo,
   przepuszcza go przez sanityzację dla każdego z 15 fontów i sprawdza, że
   wynik zawiera wyłącznie znaki obecne w `cmap` danego fontu.
2. Transkrypcja w ground truth jest **znak w znak** tym, co poszło do
   renderu (sanityzacja przed pomiarem i przed zapisem rekordu).
3. Pieczątka prostokątna o |α| ≤ 5° produkuje po jednym bboxie na linię z
   `source: "stamp"`; okrągła oraz |α| > 5° nie produkują żadnych.
4. Kreski i marker nie zmieniają liczby ani współrzędnych istniejących
   bboxów.
5. Marker nie zamazuje tekstu: po profilu `photocopy` piksele pisma pod
   paskiem pozostają wyraźnie ciemniejsze od samego paska.
6. `dataset_card.json` powstaje przy każdej generacji i zawiera commit,
   seed, komendę, listę czcionek, pełne ustawienia augmentacji oraz
   zaobserwowane statystyki; `docs/datasets.md` ma po jednym aktualnym
   wierszu na zbiór.
7. Pełny zestaw testów przechodzi, wyjście bez ostrzeżeń poza znanym
   ostrzeżeniem openpyxl.
