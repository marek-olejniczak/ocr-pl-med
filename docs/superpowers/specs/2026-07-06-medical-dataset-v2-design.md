# Generator danych v2 — nowe szablony COCO, treść medyczna, porządek w repo

Data: 2026-07-06
Status: zatwierdzony przez Tomka (sekcje 1–5 + brainstorming augmentacji)

## Cel

Dostosować pipeline generacji syntetycznych formularzy medycznych do nowego
zbioru szablonów (`dataset_30zPierwszegoPDF`, format COCO, labele
`p`/`f`/`t`/`n`/`mix`, podkłady `_blank`/`_partial`), wypełniać pola prawdziwym
polskim słownictwem medycznym (dane mają służyć i do detekcji linii, i do
treningu OCR) oraz posprzątać repozytorium.

Pipeline docelowy: wygenerowane obrazy → model detekcji linii (YOLO, kolega)
→ model OCR (kolega). Dla detekcji liczy się tylko "gdzie jest linia tekstu";
dla OCR dodatkowo "co jest napisane" — dlatego zapisujemy transkrypcję.

## 1. Wejście danych

### Nowy format (docelowy, jedyny)

Katalog szablonów zawiera `annotations.json` (COCO) + podfolder per strona:

```
templates/
  annotations.json          # COCO: images, annotations (bbox xywh, category), categories p/n/t/f/mix
  <strona>/
    <strona>.png            # oryginał z zamazanymi danymi — NIE używać
    <strona>_blank.png      # wszystkie pola poza "p" wyczyszczone — zawsze istnieje
    <strona>_partial.png    # wyczyszczone wszystko poza polami "f" — tylko gdy strona ma pola "f"
    lines/                  # wycinki z labelowania — nieużywane przez generator
```

Znaczenie labeli:
- `p` — tekst drukowany (stały element formularza); do GT zawsze, bez transkrypcji
- `f` — istniejące pismo odręczne (jawne) w oryginale
- `t` — puste pole do wypełnienia tekstem
- `n` — puste pole do wypełnienia cyframi
- `mix` — puste pole do wypełnienia mieszanką liter i cyfr

Generator (`generate_yolo_dataset.py` + `fill_form.py`) przechodzi z wejścia
CSV na loader COCO czytający powyższą strukturę. Stary loader CSV znika.

### Migracja starych danych

Jednorazowy skrypt `src/migrate_old_annotations.py`:
- wejście: `forms/segmentation/images/*.png` (13 zlabelowanych stron) +
  `C:/Users/tomek/Desktop/inzynierka/dataset/annotations.csv`
- mapowanie labeli: `printed→p`, `text→t`, `number→n` (+ stare labele pól typu
  `pesel_grid`, `city` itd. mapowane przez istniejący `LABEL_TO_CATEGORY` na
  t/n); pól `f` w starych danych nie ma
- stare szablony są już niewypełnione, więc PNG strony staje się swoim
  `_blank` (kopiowany do `templates/<strona>/<strona>_blank.png`)
- wynik dopisywany do wspólnego `templates/annotations.json`

Po migracji: 27 stron KARTA_PACJENTA + 13 starych = 40 szablonów stron,
jeden format, jeden pipeline.

## 2. Wybór podkładu per wariant

- strona ma `_partial` → 50% wariantów z `_partial`, 50% z `_blank`
- strona bez `_partial` → zawsze `_blank`

Przy `_partial`: pól `f` nie wypełniamy (już zawierają prawdziwe pismo), ale
ich bboxy trafiają do GT jako linie tekstu (`source: handwritten`, bez
transkrypcji). Przy `_blank`: pola `f` wypełniamy tak samo jak `t`.

## 3. Treść pól (zamiast pseudo-polskiego fillera)

Źródłem jest istniejąca klasa `Vocabulary` (`resources/`: common_diagnoses,
icd-10.xml, drug_names.xlsx, dosage_patterns, medical_abbreviations,
imiona/nazwiska). `generate_filler_text` zostaje zastąpione generatorem
treści per kategoria:

- **t / f**: ~70% medyczne (diagnozy, leki, dawkowanie, skróty Rp./D.S.,
  oddziały, opisy ICD, nazwy szpitali), ~30% dane osobowe (imię+nazwisko,
  miasto, adres)
- **n**: PESEL z poprawną sumą kontrolną, daty (różne formaty), telefony,
  NIP, ciągi cyfr różnej długości; kratki bez zmian — jedna cyfra per
  wykryta komórka (`detect_grid_cells`)
- **mix**: kody ICD-10 (np. J45.0), dawkowanie z jednostkami (2x500mg),
  numery dokumentów (ABC1234567)

Dopasowanie do szerokości: losujemy wyrażenie, mierzymy szerokość renderu
(z uwzględnieniem x_stretch i trackingu); za długie ucinamy po granicach
słów albo dolosowujemy krótsze. Cel wypełnienia: 30–100% szerokości pola
(bez zmian). Twardy limit 0.97 szerokości pola (bez zmian).

## 4. Wypełnianie wielolinijkowe wysokich pól

Pole, w którym mieści się ≥2 linie pisma (bbox_h ≥ ~2.2 × wysokość linii),
dostaje 1–3 linie tekstu zamiast jednej:
- liczba linii losowana z dostępnego miejsca (nie zawsze maksimum)
- odstęp między liniami ~1.2–1.6 × wysokość pisma, lekko losowy
- każda linia renderowana osobno (własna treść, własny pen fade / drift)
  i dostaje **własny bbox + własną transkrypcję** w GT
- niskie pola: zachowanie jak dotychczas (jedna linia, centrowanie /
  górna trzecia część w wysokich rubrykach)

To daje detektorowi trudny przypadek "sąsiadujące linie blisko siebie",
którego obecnie w danych nie ma wcale.

## 5. Puste pola: sumienność per formularz

Zamiast stałego `EMPTY_FIELD_PROB = 0.15` niezależnie per pole, losujemy per
formularz wskaźnik pustych pól z zakresu **0–40%** i stosujemy go do
wszystkich pól do wypełnienia. Odwzorowuje to realną korelację: jeden
formularz wypełniony w całości, inny w połowie pusty. Wartość zapisywana
w metadata.

## 6. Augmentacja skanu: trzy profile

Przed augmentacją losujemy profil (zapisywany w metadata); geometria nigdy
nie jest ruszana — bboxy pozostają co do piksela te same. Rotacji strony
NIE ma (decyzja ostateczna).

1. **Czysty skan kolorowy (~45%)** — obecne zachowanie: nierówna jasność
   0.05–0.15, szum Gaussa σ 2–6, blur 0.3–0.8, JPEG 80–92.
2. **Skala szarości (~35%)** — jak wyżej + konwersja do grayscale
   (skanery sieciowe w placówkach domyślnie skanują mono; model przestaje
   polegać na kolorze atramentu).
3. **Kserokopia (~20%)** — podbity kontrast (cienie do czerni, światła
   wypalone do bieli), szum typu sól-i-pieprz zamiast gładkiego Gaussa,
   okazjonalna pionowa smuga tonera, grayscale.

Profil "zdjęcie telefonem" odrzucony (decyzja Tomka).

## 7. Ground truth (wyjście)

Format wyjściowy: COCO `annotations.json` (jak dotychczas, kategoria
`text_line` id=1) — każda adnotacja dostaje dodatkowe pola:
- `"text"`: wygenerowana treść (dla `synthetic`), `null` dla `printed`
  i `handwritten`
- `"source"`: `printed` (boxy p z labelowania) / `synthetic` (tekst
  wygenerowany — tight bbox z atramentu) / `handwritten` (boxy f przy
  podkładzie `_partial`)

`ground_truth.csv` dostaje kolumny `source` (już jest) i `text`.
Cropy linii do treningu OCR zostaną wycięte później osobnym skryptem
z tych danych (poza zakresem tego specu).

Bez zmian: styl pisma per formularz (czcionka, rozmiar 26–40px, atrament
czarny/niebieski, WordStyle), pen fade 15%, drift ±4px, v-overflow 18%,
augmentacje per znak. Czcionki: obecne 6; rozszerzenie zbioru czcionek
po stronie Tomka, poza zakresem.

## 8. Porządek w repo

Do usunięcia z dysku (zatwierdzone, żadne nie jest śledzone w gicie):
`yolo_dataset/`, `yolo_dataset.zip`, `yolo_test/`, `training/`, `versions/`,
`font_preview.png`, `SKIEROWANIE_DO_SZPITALA.pdf`, `annotations.csv` (root),
`dataset/` (kopia starego CSV).

Git: wypisanie śledzonych `__pycache__/*.pyc` + `.gitignore`
(`__pycache__/`, `output/`, `*.zip`, artefakty generacji).

Struktura po zmianach:
- `src/` — kod (bez zmian lokalizacji)
- `resources/` — słowniki i czcionki (bez zmian)
- `forms/` — surowe PDF-y/DOC-e źródłowe (bez zmian)
- `templates/` — szablony wejściowe generatora: tylko to, czego generator
  używa (`annotations.json` + podfoldery z `_blank`/`_partial`), skopiowane
  z `dataset_30zPierwszegoPDF/` + zmigrowane stare strony
- `output/` — wszystkie generowane datasety (gitignore)
- `dataset_30zPierwszegoPDF/` — zostaje nietknięty jako surowy wynik
  labelowania (oryginały i `lines/` mogą się przydać przy etapie OCR)

## Poza zakresem (świadomie)

- rozszerzenie zbioru czcionek (Tomek zbiera)
- skreślenia / poprawki jako augmentacja (później)
- rotacja strony (nie robimy — decyzja ostateczna)
- profil "zdjęcie telefonem" (odrzucony)
- skrypt wycinający cropy linii pod OCR (następny etap)

## Kryteria sukcesu

1. `generate_yolo_dataset.py` generuje dataset z 40 szablonów (nowe + zmigrowane)
   bez błędów, z poprawnym COCO zawierającym `text` i `source`.
2. Wizualizacja (`visualize_yolo.py` / `viz/`) pokazuje tight bboxy na
   każdej wygenerowanej linii, w tym każdej linii pól wielolinijkowych
   i polach `f` z podkładów `_partial`.
3. Treść pól to czytelne polskie wyrażenia medyczne/osobowe, cyfry w
   realistycznych formatach; brak pseudo-polskich sylab.
4. Trzy profile skanu widoczne w wygenerowanym zbiorze w proporcjach
   ~45/35/20, zapisane w metadata.
5. Repo po sprzątaniu: brak wymienionych artefaktów, `.gitignore` działa,
   `git status` czysty po generacji datasetu.
