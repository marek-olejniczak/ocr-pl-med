# Plan pracy nad OCR (stan: 2026-09-10)

Punkt wyjścia: Surya-LoRA trenowana na `ocr_800k` daje CER 15,9% na testsecie
z prawdziwych notatek studenckich (baza Surya 19,9%, TrOCR-PL-FT 23,7%).
Analiza błędów LoRA (2160 linii): `k→h` 434, `a→o` 216, `t→ł` 115, `ś→s` 102;
13,6% diakrytyków źle; CER 30,8% dla GT 1–5 znaków, 21,4% dla 6–12,
~14% powyżej 25 znaków; rozrzut między autorami 6,6%–31%.

## Faza 0 — porządki (zrobione 2026-09-10)

- [x] push `augmentations` na GitHub
- [x] surowe eksporty PDF-ów i źródłowe formularze → DVC (`forms_raw/` w repo DagsHub, gałąź `dvc`)
- [x] serwis Surya w benchmarku zdejmuje HTML i tekst po `<br>` (gałąź `server`, `ecfe28d`)
- [ ] aktualny testset 2160 linii z serwera RTX 3090 (`benchmark/dane/handlabeled/`)

## Faza 1 — generator linii wycelowany w testset

Każda zmiana adresuje zmierzony błąd, nie hipotezę. Do każdej rodziny
flaga wyłączająca (potrzebna w fazie 3).

| Zmiana | Adresuje |
|---|---|
| Krótkie wycinki 3–12 znaków, pojedyncze słowa | CER 21–31% na 39% testsetu |
| Audyt fontów pod otwarte „k"/„a", dobór brakujących kształtów | `k→h`, `a→o` (~20% podmian) |
| **Sąsiednie linie wchodzące w kadr prawdziwymi literami** (opis niżej) | ciasne cropy z YOLO na gęstych notatkach |
| Erozja/dylatacja morfologiczna, ElasticTransform, GridDistortion, Downscale, blur (albumentations) | zgubione diakrytyki (13,6%), `t↔ł`, rozmycie telefonu |
| Nagłówki WIELKIMI, mieszanie wielkości | `w→W`, `p→P` |
| Kratka zeszytowa, profil „zdjęcie telefonem" | tło i oświetlenie testsetu |
| Treść anatomiczna (bazy `nlp-ner` + offline LLM „notatki z anatomii"), cyfry 22%→~3%, strzałki `->`/`=>`, wypunktowania | słownictwo i dekoder językowy |

### Sąsiednie linie w kadrze (wymaganie z 2026-09-10)

Obecny `bleed_neighbour` rysuje 2–6 krótkich losowych kresek przy górnej lub
dolnej krawędzi. To za mało: w prawdziwych cropach z YOLO od góry wchodzą
**ogonki liter z linii wyżej** (`j`, `y`, `g`, `p`, `ą`, `ę`), a od dołu
**górne części liter z linii niżej** (`l`, `t`, `d`, `k`, `ł`, `b` oraz kreski
diakrytyków `ó`, `ś`, `ź`, `ć`, `ń`).

Do zrobienia:

- renderować prawdziwy tekst sąsiada tym samym fontem i rozmiarem (styl tej
  samej ręki), przesunięty tak, żeby w kadr trafiały tylko descendery
  (linia wyżej) lub ascendery i diakrytyki (linia niżej), reszta obcięta
  przez krawędź;
- dobierać tekst sąsiada z nadreprezentacją liter z ogonkami / wysokimi
  kreskami, żeby coś faktycznie wystawało;
- losowe przesunięcie w pionie (ile wchodzi w kadr: 1–6 px) i w poziomie;
  czasem oba sąsiedzi naraz;
- prawdopodobieństwo wyższe niż dzisiejsze 18% — notatki studenckie są
  gęste; docelowo ~35–45%, do sprawdzenia w ablacji;
- GT bez zmian: tekst sąsiada nie wchodzi do etykiety;
- osobna flaga rodziny (`--no-neighbour-glyphs` lub podobna) do ablacji;
  stary `bleed_neighbour` zostaje jako wariant „kreski".

## Faza 2 — szybka weryfikacja

Jeden zbiór tej samej wielkości i z tym samym budżetem (40 000 kroków × batch 8)
co `ocr_800k`, jedna LoRA na serwerze, CER na 2160 liniach. Do tego dwa
treningi baseline z różnym seedem, żeby znać szum. Dopiero gdy jest lepiej niż
15,9% — faza 3.

## Faza 3 — ablacja

Baseline bez augmentacji + pełny + „bez jednej rodziny" × 5–6. Ten sam seed
treści, ten sam budżet kroków, CER rozbity po źródłach (autorach). Wynik:
tabela „rodzina → punkty CER". Karty zbiorów dokumentują każdy wariant.

## Faza 4 — finalny zbiór i model

Zwycięska konfiguracja → duży zbiór (pełna epoka, obecny model widział ~0,4
epoki) → finalna LoRA → benchmark → aplikacja.

## Faza 5 — dokumenty (jeśli zostanie czas)

Elastic na całych stronach z przeliczaniem bboxów, LLM dla spójności
formularza. Detektor działa dobrze, więc najniższy priorytet.
