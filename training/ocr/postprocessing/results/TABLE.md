# Tabela wyników (generowana: `python -m evaluation.make_table`)

Predykcje out-of-fold, zagnieżdżone LOGO, 2160 linii handlabeled. ΔCER względem baseline swojego wiersza. Wiersze „hist.” dotyczą predykcji z 2026-10-01 i starszego GT; oracle i opcja 0 także.

| wariant | CER | CER pomocn. | WER | ΔCER p.p. | Δ wzgl. | 95% CI grupy | 95% CI linie | p (znaki) | fixed / broken / ch-wrong / net | prec. | recall | harm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (Surya v2_200k, dataset-v1) | 0.12701 | 0.11741 | 0.4831 | — | — | — | — | — | — | — | — | — |
| opcja 0 (higiena) | 0.12769 | 0.11953 | 0.4817 | +0.00 | — | — | — | — | 0 / 0 / 0 / 0 | — | — | 0 |
| **główny: dataset-v1, lista TA en,la,pl, progi strojone bez listy** | 0.09862 | 0.08768 | 0.3552 | -2.84 | -22.4% | [-3.40, -1.89] | [-3.06, -2.62] | <1e-6 | 780 / 10 / 143 / 770 | 0.836 | 0.278 | 0.31% |
| dataset-v1, bez listy | 0.09902 | 0.08809 | 0.3574 | -2.80 | -22.0% | [-3.37, -1.87] | [-3.02, -2.58] | <1e-6 | 780 / 23 / 143 / 757 | 0.825 | 0.278 | 0.71% |
| dataset-v1, lista TA la,pl | 0.09881 | 0.08787 | 0.3566 | -2.82 | -22.2% | [-3.37, -1.89] | [-3.03, -2.60] | <1e-6 | 780 / 18 / 143 / 762 | 0.829 | 0.278 | 0.56% |
| ablacja dataset-v1: lista także w strojeniu (luźniejsze progi) | 0.09602 | 0.08495 | 0.3440 | -3.10 | -24.4% | [-3.74, -2.06] | [-3.33, -2.86] | <1e-6 | 851 / 13 / 225 / 838 | 0.781 | 0.304 | 0.40% |
| hist.: opcja 1, predykcje z 1.10, GT po poprawce; baseline 0.12727 | 0.09989 | 0.09021 | 0.3556 | -2.74 | -21.5% | [-3.34, -1.81] | [-2.94, -2.52] | <1e-6 | 763 / 17 / 163 / 746 | 0.809 | 0.273 | 0.53% |
| hist.: opcja 1, predykcje z 1.10, GT przed poprawką; baseline 0.12769 | 0.10483 | 0.09554 | 0.3773 | -2.29 | -17.9% | [-2.93, -1.48] | [-2.47, -2.09] | <1e-6 | 642 / 13 / 117 / 629 | 0.832 | 0.228 | 0.40% |
| hist.: harm ≤ 0.5% | 0.09945 | 0.08975 | 0.3523 | -2.82 | -22.1% | [-3.46, -1.83] | [-3.05, -2.59] | <1e-6 | 806 / 26 / 214 / 780 | 0.771 | 0.287 | 0.81% |
| hist. ablacja: bez jednostek wieloznakowych | 0.11112 | 0.10218 | 0.3962 | -1.66 | -13.0% | [-2.28, -0.97] | [-1.84, -1.49] | <1e-6 | 531 / 16 / 115 / 515 | 0.802 | 0.189 | 0.50% |
| hist. ablacja: bez klas chronionych P2–P8 | 0.10483 | 0.09554 | 0.3773 | -2.29 | -17.9% | [-2.94, -1.48] | [-2.47, -2.10] | <1e-6 | 642 / 13 / 119 / 629 | 0.829 | 0.228 | 0.40% |
| hist.: opcja 2 osobno (tylko diakrytyki) | 0.12675 | 0.11856 | 0.4751 | -0.09 | -0.7% | [-0.15, -0.05] | [-0.13, -0.06] | 1.0e-06 | 53 / 13 / 4 / 40 | 0.757 | 0.019 | 0.40% |
| hist. ablacja: + sjp odmiany | 0.10637 | 0.09712 | 0.3848 | -2.13 | -16.7% | [-2.86, -1.20] | [-2.33, -1.95] | <1e-6 | 604 / 20 / 129 / 584 | 0.802 | 0.215 | 0.62% |
| hist. ablacja: forward zamiast Viterbi | 0.10450 | 0.09520 | 0.3760 | -2.32 | -18.2% | [-2.98, -1.48] | [-2.51, -2.12] | <1e-6 | 656 / 19 / 115 / 637 | 0.830 | 0.233 | 0.59% |
| hist.: test odporności, podział gruby (6 grup), harm ≤ 0.3% | 0.10266 | 0.09327 | 0.3616 | -2.50 | -19.6% | [-3.23, -0.94] | [-2.73, -2.27] | <1e-6 | 769 / 45 / 276 / 724 | 0.706 | 0.273 | 1.40% |
| hist.: podział gruby, harm ≤ 0.5% | 0.09613 | 0.08622 | 0.3370 | -3.16 | -24.7% | [-3.76, -1.47] | [-3.39, -2.89] | <1e-6 | 921 / 49 / 350 / 872 | 0.698 | 0.327 | 1.53% |
| oracle: OOV → GT (GT w słowniku, lev ≤ 2) | 0.08020 | 0.07220 | 0.2495 | -4.75 | -37.2% | — | — | — | 1399 / 0 / 0 / 1399 | 1.000 | 0.497 | 0 |
| oracle: OOV → GT (GT w słowniku, lev ≤ 3) | 0.06066 | 0.05234 | 0.2030 | -6.70 | -52.5% | — | — | — | 1679 / 0 / 0 / 1679 | 1.000 | 0.597 | 0 |
| oracle: diakrytyki | 0.12410 | 0.11583 | 0.4568 | -0.36 | -2.8% | — | — | — | 150 / 0 / 0 / 150 | 1.000 | 0.053 | 0 |

## Aneks: CER per grupa

### podział fine

| wariant | G1_noga | G2_anatomia_zmysly | G3_serce | G4_unaczynienie | G5_elementy_topo | G6_dataN | G7_zrzuty_0507 | G8_zrzuty_0512 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.1173 | 0.2182 | 0.0939 | 0.0484 | 0.1133 | 0.2211 | 0.1371 | 0.1245 |
| **główny: dataset-v1, lista TA en,la,pl, progi strojone bez listy** | 0.0852 | 0.2036 | 0.0562 | 0.0457 | 0.0771 | 0.2039 | 0.1089 | 0.0880 |
| hist.: opcja 1, predykcje z 1.10, GT przed poprawką; baseline 0.12769 | 0.0961 | 0.2270 | 0.0538 | 0.0426 | 0.0787 | 0.2128 | 0.1146 | 0.0936 |
| hist.: harm ≤ 0.5% | 0.0830 | 0.2233 | 0.0608 | 0.0423 | 0.0715 | 0.2122 | 0.1063 | 0.0863 |
| hist. ablacja: bez jednostek wieloznakowych | 0.1079 | 0.2299 | 0.0631 | 0.0431 | 0.0867 | 0.2175 | 0.1207 | 0.0977 |
| hist. ablacja: bez klas chronionych P2–P8 | 0.0961 | 0.2273 | 0.0538 | 0.0426 | 0.0787 | 0.2128 | 0.1146 | 0.0935 |
| hist.: opcja 2 osobno (tylko diakrytyki) | 0.1199 | 0.2350 | 0.0881 | 0.0452 | 0.1028 | 0.2258 | 0.1335 | 0.1239 |
| hist. ablacja: + sjp odmiany | 0.1080 | 0.2264 | 0.0538 | 0.0441 | 0.0859 | 0.2159 | 0.1104 | 0.0935 |
| hist. ablacja: forward zamiast Viterbi | 0.0961 | 0.2270 | 0.0538 | 0.0426 | 0.0787 | 0.2128 | 0.1146 | 0.0921 |

### podział coarse

| wariant | C_zrzuty_dataN | G1_noga | G2_anatomia_zmysly | G3_serce | G4_unaczynienie | G5_elementy_topo |
|---|---|---|---|---|---|---|
| baseline | 0.1381 | 0.1213 | 0.2356 | 0.0904 | 0.0452 | 0.1060 |
| hist.: test odporności, podział gruby (6 grup), harm ≤ 0.3% | 0.1034 | 0.1052 | 0.2290 | 0.0631 | 0.0434 | 0.0787 |
| hist.: podział gruby, harm ≤ 0.5% | 0.0988 | 0.0855 | 0.2259 | 0.0624 | 0.0402 | 0.0723 |

### podział gtfix

| wariant | G1_noga | G2_anatomia_zmysly | G3_serce | G4_unaczynienie | G5_elementy_topo | G6_dataN | G7_zrzuty_0507 | G8_zrzuty_0512 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.1207 | 0.2367 | 0.0900 | 0.0449 | 0.1060 | 0.2248 | 0.1334 | 0.1238 |
| hist.: opcja 1, predykcje z 1.10, GT po poprawce; baseline 0.12727 | 0.0870 | 0.2282 | 0.0513 | 0.0415 | 0.0755 | 0.2112 | 0.1063 | 0.0891 |

