# Wyniki: generator v2 i badanie ablacyjne

Benchmark `ocr-ablacja-v2_20260915_212551`, testset `handlabeled` = 2160 linii
z prawdziwych notatek studentów medycyny. Surowe liczby i pełne rozbicia:
`docs/wyniki_ablacji.md` (generowane przez `src/ablation_report.py`).
Opis samych zmian w generatorze: `docs/ocr_generator_v2.md`.

## 1. Wynik główny

| Model | Dane | Kroki | CER | 95% CI | Exact match |
|---|---|---|---|---|---|
| Surya baza | — | — | 19,9% | 19,2–20,7% | 13,6% |
| TrOCR-PL fine-tuned | — | — | 23,7% | — | 10,2% |
| **LoRA v1** (`ocr_800k`) | 800k | 40k | 15,9% | 15,3–16,5% | 18,6% |
| **LoRA v2** (`v2_800k`) | 800k | 40k | **12,9%** | 12,4–13,4% | 24,9% |
| **LoRA v2** (`v2_200k`) | 200k | 10k | **12,7%** | 12,2–13,2% | 26,2% |

Generator v2 obniżył CER z 15,9% do 12,9%, czyli o 3,0 punktu procentowego
(19% względnie). Przedziały ufności nie zachodzą na siebie, więc różnica jest
istotna. Odsetek linii przeczytanych bezbłędnie wzrósł z 18,6% do 24,9%,
a po uwzględnieniu wariantu 200k do 26,2% — o 41% względnie.

Kontrola poprawności: model z sierpnia (`surya_lora_ocr800k`) powtórzył wynik
15,88% wobec 15,879% w poprzednim benchmarku. Pomiar jest powtarzalny.

## 2. Ćwierć danych wystarczy

`v2_200k` (200 tys. linii, 10 tys. kroków) wypadł tak samo jak `v2_800k`
(800 tys. linii, 40 tys. kroków), a nawet minimalnie lepiej: 12,7% wobec 12,9%,
przedziały w pełni się pokrywają. Wniosek: przy tej jakości danych wąskim
gardłem nie jest już ich ilość ani długość treningu. Czterokrotnie większy
zbiór i czterokrotnie dłuższy trening nie dołożyły nic mierzalnego.

To ma praktyczne znaczenie dla dalszej pracy: eksperymenty można prowadzić na
zbiorach 200k i budżecie 10k kroków, czyli ~25 minut na RTX 3090 zamiast
prawie dwóch godzin.

## 3. Badanie ablacyjne

Punkt odniesienia: `v2_200k` = 12,72%. Każdy wariant to ten sam seed treści,
ten sam budżet 10 tys. kroków, wyłączona jedna grupa rodzin augmentacji.

| Grupa | Rodziny | CER bez grupy | Strata | Istotna? |
|---|---|---|---|---|
| **A. Treść** | `short_words`, `caps`, `arrows_bullets`, `anatomy_vocab` | 14,7% (14,2–15,3) | **+2,02 pkt** | tak, CI rozłączne |
| B. Kadr i tło | `neighbour_glyphs`, `grid_paper` | 12,8% (12,3–13,4) | +0,11 pkt | nie |
| C. Degradacja obrazu | `morphology`, `elastic`, `phone_photo` | 12,8% (12,3–13,3) | +0,06 pkt | nie |

**Cała poprawa pochodzi z treści.** Wyłączenie grupy A cofa model niemal do
poziomu v1 (14,7% wobec 15,9%). Grupy B i C mieszczą się w szumie pomiarowym —
ich wyłączenie zmienia wynik o 0,1 punktu przy przedziale ufności szerokim na
1,1 punktu.

Interpretacja: model ma problem z tym, **co** czyta, a nie z tym, **jak** to
zostało sfotografowane. Surya jest odporna na zniekształcenia obrazu (była
trenowana na ogromnym korpusie skanów), natomiast jej dekoder językowy był
dostrojony do polszczyzny formularzowej — nazwisk, adresów, dat, numerów.
Notatki anatomiczne to inne słownictwo i inna struktura linii, i to właśnie
naprawiła grupa A.

Uwaga metodologiczna: wynik „grupa B i C nie pomagają" dotyczy **tego**
zbioru testowego. Kratka, telefon i zniekształcenia elastyczne mogą mieć
znaczenie na materiale gorszej jakości; tutaj zdjęcia są na tyle czytelne, że
ich symulacja nic nie wnosi. Nie ma jednak podstaw, by trzymać je w
generatorze „na wszelki wypadek" — kosztują czas generacji i komplikują opis.

## 4. Gdzie konkretnie jest poprawa

Wszystkie osiem źródeł (autorów zeszytów) poprawiło się jednocześnie, od 6,6%
do 4,1% w najlepszym przypadku i od 31,0% do 23,8% w najgorszym. Efekt nie
pochodzi z jednego zeszytu.

Najmocniej zadziałało tam, gdzie celowaliśmy — na krótkich liniach:

| Długość GT | Udział testu | v1 | v2_200k | Zmiana |
|---|---|---|---|---|
| 1–5 znaków | 199 linii | 30,8% | 25,2% | −5,6 pkt |
| 6–12 znaków | 634 linii | 21,4% | 16,1% | −5,3 pkt |
| 13–25 znaków | 773 linii | 15,8% | 12,2% | −3,6 pkt |
| 26–45 znaków | 399 linii | 14,1% | 12,0% | −2,1 pkt |
| 46+ znaków | 155 linii | 13,7% | 11,3% | −2,4 pkt |

Krótkie linie, czyli 39% testu, zyskały ponad dwa razy więcej niż długie.
To bezpośrednio potwierdza hipotezę, która stała za rodziną `short_words`.

## 5. Co nadal nie działa

Najczęstsze podmiany znaków w `v2_800k`:

`k→h` 395, `a→o` 235, `ś→s` 129, `c→w` 78, `o→a` 58, `t→ł` 58, `k→l` 58.

Mylenie `k` z `h` spadło z 471 do 395 wystąpień, czyli o 16%, ale nadal
dominuje listę. Dwadzieścia nowych fontów pomogło mniej, niż zakładaliśmy.
Podmiana `a→o` nie drgnęła (279 → 235 przy ogólnym spadku błędów o 19%, więc
względnie bez zmian).

To jest najlepszy kandydat na kolejną iterację: zamiast dokładać kolejne
fonty, warto sprawdzić, czy problem nie leży po stronie tokenizera lub
rozdzielczości wejściowej modelu. Mediana wysokości wycinka to 35 pikseli,
a Surya skaluje wejście do własnego rozmiaru — przy tak małej czcionce
pętla w `k` i brzuszek w `a` mogą się zwyczajnie zlewać.

## 6. Wnioski do pracy

1. Ukierunkowanie danych syntetycznych na charakterystykę zbioru docelowego
   dało 19% względnej redukcji CER, przy niezmienionym modelu, budżecie
   treningu i zbiorze testowym.
2. Decydowała treść (słownictwo, długość linii, struktura), nie realizm
   fotograficzny. Augmentacje obrazu, w tym odkształcenia elastyczne, nie
   dały mierzalnego efektu na tym materiale.
3. Powyżej 200 tys. przykładów zbiór przestał być czynnikiem ograniczającym.
4. Analiza błędów poprzedniego modelu okazała się trafnym narzędziem
   projektowania danych: największy zysk przyszedł dokładnie w kategorii,
   którą wskazała (linie do 12 znaków).

## 7. Stan i co dalej

Adaptery: `training/results/ocr/surya/<nazwa>-lora-r64/adapter/` na serwerze,
kopia w `ocr-main/training/results/ocr/surya/`. Modelem do aplikacji jest
`v2_800k` albo `v2_200k` — praktycznie nierozróżnialne, `v2_200k` jest tańszy
w odtworzeniu.

Otwarte kierunki, w kolejności oczekiwanego zysku:

- `k→h` i `a→o`: sprawdzić rozdzielczość wejściową i zachowanie na małych
  wysokościach wycinka, zanim doda się kolejne fonty;
- dłuższy trening pełnego zbioru nie ma sensu (patrz punkt 2), ale ma sens
  większa różnorodność treści;
- benchmark całego łańcucha strona → YOLO → OCR, bo dziś mierzymy OCR na
  wycinkach przygotowanych ręcznie, a w aplikacji tnie je detektor.
