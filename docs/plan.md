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

## Faza 1 — generator linii wycelowany w testset (kod: 2026-09-10)

Każda zmiana adresuje zmierzony błąd, nie hipotezę. Każda rodzina ma flagę
`--disable <nazwa>` w `generate_ocr_lines.py` (potrzebne w fazie 3); `--disable all`
daje stary renderer. Arkusze próbek: `python src/preview_line_families.py`.

| Rodzina (`--disable`) | Zmiana | Adresuje | Stan |
|---|---|---|---|
| `short_words` | pojedyncze słowa 3–12 znaków (34% linii), czasem dwa | CER 21–31% na 39% testsetu | zrobione |
| (fonty) | audyt: dodane 20 fontów Google z polskimi znakami, w tym szkolne Playwrite (k z pętlą), 35 fontów razem | `k→h`, `a→o` | zrobione |
| `neighbour_glyphs` | prawdziwe litery sąsiednich linii w kadrze (opis niżej), 40% | ciasne cropy z YOLO | zrobione |
| `morphology` | erozja 2×2 z mieszaniem / dylatacja 1–2 px, 30% | zgubione diakrytyki, `t↔ł` | zrobione |
| `elastic` | ElasticTransform / GridDistortion (albumentations), 30% | wygięta kartka, drżenie ręki | zrobione |
| `caps` | nagłówki WIELKIMI (8% linii) + wielka litera na początku (12%) | `w→W`, `p→P`; 7% znaków testsetu to wersaliki | zrobione |
| `grid_paper` | kratka / linie zeszytowe, 45% | tło testsetu | zrobione |
| `phone_photo` | cień, balans bieli, rozmycie, downscale, mocny JPEG, 40% | zdjęcie telefonem zamiast skanu | zrobione |
| `arrows_bullets` | `- `, `1) `, `a) `, `->`, `=>` | 7% linii testsetu z wypunktowaniem, 4% ze strzałką | zrobione |
| `anatomy_vocab` | pule: `anatomy_seed.txt` (ręcznie), `anatomy_phrases.txt` (Ollama gemma4), ICD-11, badania lab., zabiegi; cyfry 22%→~3% | słownictwo i dekoder językowy | zrobione |

Do sprawdzenia przed fazą 2: jakość puli z LLM (gemma4 e4b miesza słowa,
dostała mniejszą wagę niż lista ręczna).

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

## Faza 2 — weryfikacja (ZROBIONE 2026-09-15)

Generator v2, 800k linii, 40k krokow: **CER 12,9%** wobec 15,9% dla v1
(przedzialy ufnosci rozlaczne). Exact match 18,6% -> 24,9%. Pelne omowienie:
`docs/wyniki_ocr_v2.md`, liczby: `docs/wyniki_ablacji.md`.

## Faza 3 — ablacja (ZROBIONE 2026-09-15)

Punkt odniesienia `v2_200k` = 12,72%. Wynik:

| Grupa | CER bez grupy | Strata | Istotna? |
|---|---|---|---|
| A. Tresc (short_words, caps, arrows_bullets, anatomy_vocab) | 14,7% | +2,02 pkt | tak |
| B. Kadr i tlo (neighbour_glyphs, grid_paper) | 12,8% | +0,11 pkt | nie |
| C. Degradacja (morphology, elastic, phone_photo) | 12,8% | +0,06 pkt | nie |

Cala poprawa pochodzi z tresci. Augmentacje obrazu, w tym elastic, nie daly
mierzalnego efektu na tym testsecie. Dodatkowo: `v2_200k` (200k linii, 10k
krokow) rowny `v2_800k` (800k, 40k), wiec zbior przestal byc czynnikiem
ograniczajacym.

## Faza 4 — finalny zbiór i model

Zwycięska konfiguracja → duży zbiór (pełna epoka, obecny model widział ~0,4
epoki) → finalna LoRA → benchmark → aplikacja.

## Faza 5 — dokumenty (jeśli zostanie czas)

Elastic na całych stronach z przeliczaniem bboxów, LLM dla spójności
formularza. Detektor działa dobrze, więc najniższy priorytet.
