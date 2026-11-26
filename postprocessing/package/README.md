# ocr-postprocess: korekta po OCR

Opcjonalny krok za Suryą. Dostaje linie tekstu z OCR i zwraca je poprawione: słowo spoza słownika zamienia na
słowo słownikowe, gdy model pomyłek Suryi i częstość słowa dają wyraźną przewagę. Nie zmienia słów słownikowych,
liczb, skrótów ani terminów anatomicznych. Nie dodaje brakujących słów i nie łączy rozbitych.

Wszystko, czego korektor się nauczył, jest w dołączonej paczce `v4_200k_25k` (kilka KB). W aplikacji nic się nie
trenuje.

## 1. Instalacja

Wymaga Pythona ≥ 3.10. Z katalogu głównego repo:

```bash
pip install "./postprocessing/package[surya]"   # z oceną Suryi (docelowo); bez [surya] sam tryb statyczny
ocr-postprocess setup                           # raz: słowniki + kompilacja indeksu + test
```

`setup` pobiera dwa słowniki (ok. 70 MB, sprawdzane SHA-256), kompiluje indeks (ok. 40–70 s) i poprawia linię
kontrolną. Kończy się kodem 0 tylko wtedy, gdy korektor działa. Potrzebuje internetu; potem korektor działa offline.

| co | ile |
|---|---|
| miejsce na dysku po `setup` | ok. 420 MB w `$PP_CACHE` (domyślnie `~/.cache/ocr-pl-med/postprocessing`) |
| start korektora | ok. 1.5 s (bez skompilowanego indeksu ok. 23 s) |
| RAM po załadowaniu | ok. 1.6 GB (obok modelu Surya) |
| strona 30 linii, tryb statyczny, CPU | ok. 1.5 s |

`$PP_CACHE` musi wskazywać to samo miejsce przy `setup` i przy działaniu aplikacji (ważne przy instalatorze,
kontenerze albo innym użytkowniku systemu). Bez indeksu pierwsze `load()` zbuduje go samo; bez słowników zgłosi
błąd z poleceniem `ocr-postprocess setup`.

`[surya]` przypina `surya-ocr==0.17.1` i `transformers==4.56.1`, czyli wersje, na których trenowano adapter.

## 2. Podłączenie (tryb docelowy: z oceną Suryi)

Działający przykład do skopiowania: [`examples/surya_worker.py`](examples/surya_worker.py). Najważniejsze:

```python
from peft import PeftModel
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from ocr_postprocess import load
from ocr_postprocess.surya_score import SuryaScorer

foundation = FoundationPredictor()
foundation.model = PeftModel.from_pretrained(foundation.model, ADAPTER).eval()
recognition = RecognitionPredictor(foundation)      # OCR
scorer = SuryaScorer(foundation)                    # ocena kandydatów: TEN SAM model, raz w pamięci
corrector = load()                                  # raz, przy starcie (można w osobnym wątku)

lines = ...                                         # tekst z OCR dla wszystkich linii strony
fixed = corrector.correct_lines_detailed(lines, images=crops, scorer=scorer)
# [{"text": "WZGÓRZE - impulsy z", "changes": [{"token": 0, "from": "WZGORZE", "to": "WZGÓRZE"}]}, ...]
```

Po stronie aplikacji:

1. **Jeden `FoundationPredictor`** dla OCR i oceny. Drugi egzemplarz to podwójne zużycie pamięci.
2. **Ten sam wycinek linii** (`PIL.Image`) do OCR i do oceny, w tej samej kolejności co teksty. Jeden obraz = jedna linia.
3. **Cała strona jednym wywołaniem.** Wyszukiwanie kandydatów jest szybkie w partii: ok. 11 ms na słowo, a dla
   pojedynczego słowa ok. 340 ms (pomiar bez skompilowanego indeksu).
4. **Czyszczenie tekstu jak w przykładzie** (`clean`): ucięcie po `<br>`, usunięcie tagów, encje HTML. Na tak
   czyszczonym tekście mierzono wyniki. Białe znaki i zapętlenia modelu korektor czyści sam.
5. **Nie wołać OCR i oceny równolegle** na jednym modelu.
6. **Błąd korekty nie może zatrzymać OCR:** ocena Suryi → statyczny → surowy tekst (tak robi przykład).
7. Opcjonalnie podświetlić w interfejsie słowa z `changes` (`token` = indeks po podziale po spacjach) i dać
   użytkownikowi podgląd tekstu sprzed korekty. Korektor czasem zamienia błąd OCR na inne, wiarygodnie wyglądające słowo.

### Tryb statyczny (zapasowy)

Gdy nie ma obrazu linii albo modelu: `corrector.correct_lines(lines)` albo `correct_lines_detailed(lines)`.
Bez zależności od Suryi i torcha. Dostępny też z wiersza poleceń i przez HTTP (dla aplikacji nie w Pythonie):

```bash
printf 'WZGORZE - impulsy z\n' | ocr-postprocess correct
ocr-postprocess serve --port 5056 --preload
curl -s localhost:5056/health
curl -s -X POST localhost:5056/correct -d '{"lines": ["WZGORZE - impulsy z"], "detailed": true}'
```

Serwis HTTP obsługuje **tylko** tryb statyczny i nasłuchuje na 127.0.0.1.

## 3. Czego się spodziewać

Zmierzone na Surya `v4_200k_25k` i ręcznie etykietowanym test/real/v1.2 (2904 linie odręczne, 11 źródeł):

| tryb | CER, leave-one-group-out | CER, zamrożony test | harm |
|---|---|---|---|
| bez korekty | 0.1098 | 0.1040 | — |
| statyczny | 0.0939 (−1.59 p.p.) | 0.0887 (−1.53) | 0.32% / 0.20% |
| z oceną Suryi | 0.0858 (−2.40 p.p.) | 0.0784 (−2.56) | 0.40% / 0.31% |

Harm = odsetek słów poprawnych przed korektą, które korektor zepsuł. Pełny opis i zastrzeżenia:
`../results/v12/README.md`. Liczby zostaną przeliczone na finalnym zbiorze testowym.

- Zysk zmierzono głównie na **notatkach anatomicznych**. Na kartach pacjenta (141 linii) korektor prawie nic nie
  zmienia: liczby, nazwiska, daty i skróty dawkowania są poza jego zakresem. Na angielskich nazwach leków brak zysku.
- Paczka jest **związana z adapterem OCR**: model pomyłek i progi pochodzą z błędów `v4_200k_25k`. Z innym adapterem
  będzie działać, ale gorzej i bez zmierzonej jakości. Nową paczkę buduje `../experiments/` (`rebuild.py`);
  wczytanie: `load("nazwa")` albo `load("/sciezka/do/paczki")`.
- Lista chronionych terminów obejmuje łacinę, angielski i polski (`en,la,pl`).

## 4. Platformy

Pakiet nie wybiera urządzenia: tryb z oceną działa tam, gdzie Surya załaduje model (autodetekcja albo zmienna
`TORCH_DEVICE=cuda|mps|cpu`). Tryb statyczny używa tylko CPU.

**Na każdej nowej maszynie uruchom raz:**

```bash
ocr-postprocess check-surya --adapter KATALOG_ADAPTERA
```

Test rysuje linię tekstu i sprawdza, czy ocena liczona w partii zgadza się z liczoną pojedynczo i czy przekręcony
tekst dostaje niższą ocenę. Wypisuje też szybkość. Przy błędzie zostań przy trybie statycznym.

Co sprawdzono (2026-11-25/26, adapter `v4_200k_25k`):

| maszyna | zgodność | OCR, na linię | ocena Suryi, wariantów/s | dodatek na stronę 30 linii (ok. 150 wariantów) |
|---|---|---|---|---|
| RTX 3090, bf16 | wzorzec (na nim mierzono wyniki) | — | ok. 100 (przed przyspieszeniem) | ok. 1.5 s |
| GTX 1650 4 GB, fp16 | OK; oceny różne od 3090 o maks. 0.52 nat | ok. 1.5 s | 2.6 | 33 s zmierzone na stronie 30 linii (OCR: 44 s) |
| CPU laptopa (12 wątków), float32 | OK (`check-surya`) | 3.4 s | 1.7 | ok. 90 s (szacunek z 10 linii) |
| Mac, MPS | **niesprawdzone**: uruchom `check-surya` | | | |
| Windows | **niesprawdzone** (tak samo tryb statyczny) | | | |

Wniosek: tryb z oceną jest szybki na mocnym GPU. Na słabym GPU i na CPU dokłada mniej więcej tyle co sam OCR
(CPU) albo 3/4 tego czasu (GTX 1650); jeśli to za dużo, użyj statycznego (ok. 1.5 s na stronę, −1.59 zamiast
−2.40 p.p.). Wygodny układ: pokazać od razu tekst z korektą statyczną, a ocenę Suryi policzyć w tle i podmienić.

Mało pamięci GPU: `SuryaScorer(foundation, max_batch=4)`; pomaga też
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (na karcie 4 GB mierzono z `max_batch=8` i tą zmienną).

## 5. Problemy

| objaw | przyczyna i rozwiązanie |
|---|---|
| `brak …/polimorf-…: uruchom ocr-postprocess setup` | nie uruchomiono `setup` albo aplikacja ma inny `$PP_CACHE` / użytkownika |
| start trwa kilkadziesiąt sekund | brak skompilowanego indeksu w `$PP_CACHE/compiled/`: uruchom `setup` |
| `brak progów z oceną Suryi` | paczka zbudowana bez `--params-rescored`; użyj trybu statycznego albo dołączonej paczki |
| `CUDA out of memory` przy ocenie | mniejsze `max_batch` i `expandable_segments` (§4) |
| `N linii, ale M obrazów` | obrazy i teksty muszą być parami, w tej samej kolejności |
| pobieranie słownika kończy się 404 | wydawca usunął przypiętą wersję: nowy URL i SHA-256 w `resources.py`; indeks przebuduje się sam |

## Licencje

- PoliMorf (2026-10-18), Zespół Inżynierii Lingwistycznej IPI PAN / SGJP, BSD 2-clause.
- NKJP 1-gramy (podkorpus zrównoważony), IPI PAN, CC-BY.
- Lista terminów anatomicznych: etykiety z Wikidata, CC0.
- Skróty anatomiczne: Terminologia Anatomica (TA98), ręczna lista.

Słowniki nie są częścią repo ani pakietu; pobiera je `setup`. Rozpowszechniając je (np. w instalatorze razem z
indeksem), trzeba dołączyć powyższe noty. Licencji samej Suryi i jej wag ten pakiet nie obejmuje.

## Układ i testy

| plik w `src/ocr_postprocess/` | co |
|---|---|
| `bundle.py`, `models/` | paczki (model pomyłek + progi), `load()` |
| `correct.py`, `decide.py` | API `Corrector`, reguła decyzji |
| `candidates.py`, `error_model.py` | kandydaci (Levenshtein w partii), model kanału Brill & Moore |
| `lexicon.py`, `lm.py`, `compiled.py` | słownik, częstości słów, skompilowany indeks |
| `normalize.py`, `tokenize.py`, `protect.py` | czyszczenie, tokeny, klasy chronione |
| `surya_score.py`, `surya_check.py` | ocena kandydatów przez Suryę, test zgodności |
| `resources.py`, `__main__.py`, `server.py` | pobieranie słowników, polecenia, HTTP |
| `data/ta_terms.tsv` | chronione terminy anatomiczne |

```bash
pip install pytest && pytest -q postprocessing/package/tests      # nie wymaga słowników ani Suryi
```
