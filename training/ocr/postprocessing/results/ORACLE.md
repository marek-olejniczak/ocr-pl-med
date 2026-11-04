# Oracle: górne granice zysku (P1, 2026-11-04)

`PYTHONPATH=. python -m evaluation.oracle` → `oracle.csv`, `oracle_per_group.csv`,
`gt_diacritics_by_group.csv`, `suspect_gt_diacritics.csv`.

Oracle podmienia token predykcji na token GT (wyrównanie 1:1 na słowach), gdy spełnia warunek.
Nie łączy i nie dzieli tokenów. „OOV” = rdzeń tokenu (bez brzegowej interpunkcji, małe litery,
same litery/dywiz) spoza PoliMorf 2026-10-18 (5.35 mln form, łącznie z członami haseł
wieloczłonowych, jak w tickecie 05).

| wariant | CER | ΔCER p.p. | WER | tokenów |
|---|---|---|---|---|
| baseline | 0.12769 | — | 0.48174 | — |
| O0 higiena (NFC, zero-width, spacje) | 0.12769 | 0.00 | 0.48174 | 0 |
| O1 każdy błędny token → GT | 0.02546 | −10.22 | 0.03154 | 2712 |
| O1 OOV → GT | 0.03680 | −9.09 | 0.11952 | 2182 |
| O1 OOV → GT, GT w słowniku, lev ≤ 3 | 0.06065 | −6.70 | 0.20302 | 1679 |
| O1 OOV → GT, GT w słowniku, lev ≤ 2 | 0.08020 | −4.75 | 0.24950 | 1399 |
| O1 OOV → GT, GT w słowniku, lev ≤ 1 | 0.10701 | −2.07 | 0.34246 | 839 |
| O2 wszystkie błędy tylko-diakrytyczne → GT | 0.12410 | −0.36 | 0.45684 | 150 |
| O2 jw., tylko OOV | 0.12514 | −0.26 | 0.46414 | 106 |
| metryka: diakrytyki zdjęte po obu stronach | 0.11769 | −1.00 | — | — |
| metryka pomocnicza (casefold, bez interpunkcji) | 0.11953 | −0.82 | 0.44367 | — |

## Wnioski

- **Opcja 0 jako korekta nic nie daje**: predykcje są już w NFC, bez znaków zero-width i nadmiarowych
  spacji. Zostaje tylko jako metryka pomocnicza (−0.82 p.p., tylko w raporcie).
- **Opcja 1 ma duży zapas**: korekta OOV z kandydatem w słowniku i w zasięgu lev ≤ 2 to pułap −4.75 p.p.
  (lev ≤ 3: −6.70). 20% błędnych tokenów (530 z 2712) to formy istniejące w słowniku (real-word), poza zasięgiem
  reguły „tylko OOV” (−10.22 vs −9.09). Realny korektor zdobędzie ułamek pułapu, bo musi wybrać
  właściwego kandydata; to mierzy P4.
- **Opcja 2 osobno jest mała**: pułap −0.36 p.p. (150 tokenów), tylko OOV: −0.26 p.p. Spadek CER
  o 1.0 p.p. przy zdjęciu diakrytyków z metryki to w większości tokeny, które mają też inne błędy.
  Potwierdza decyzję, żeby wchłonąć opcję 2 w model kanału.
- Przedziały po grupach są 2–4× szersze niż po liniach (np. lev ≤ 2: [−5.36, −4.06] vs
  [−5.00, −4.49] p.p.).
- Rozrzut między grupami jest duży (CER baseline 0.045–0.236). G2 anatomia_zmysly i G6 dataN
  mają najwięcej błędów spoza zasięgu lev ≤ 3.

## Konwencja diakrytyków w GT (ticket 09)

101 tokenów GT (1.81% słów) są spoza słownika, a ich wariant z diakrytykami jest w słowniku.
Skupiają się w zrzutach (G7 2.90%, G8 3.63%; pozostałe grupy ≤ 1.8%), więc to raczej
cecha etykietowania tych źródeł niż pisma. W 26 przypadkach **Surya przewidział wariant
z diakrytykiem**, a metryka liczy to jako błąd. Lista zawiera też fałszywe alarmy (skróty jak
„śr”, dwuznaczne „cos/coś”), stąd ręczny przegląd. Oficjalny GT zostaje bez zmian (decyzja
2026-11-04); po przeglądzie raportujemy wynik także na GT z poprawionymi etykietami.
