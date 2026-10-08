## Koszt kazdego warunku (zbior kontrolny, pary)

Zmiana CER wzgledem czystego renderu tych samych linii; 95% przedzial bootstrap po liniach bazowych.

| warunek | rodzina | sila | v1 CER | v1 zmiana | v2 CER | v2 zmiana |
|---|---|---|---|---|---|---|
| clean | none | 0 | 1.6% | +0.0 (+0.0..+0.0) | 0.6% | +0.0 (+0.0..+0.0) |
| neighbours_s1 | neighbour_glyphs | 1 | 1.7% | +0.1 (-0.0..+0.3) | 0.7% | +0.1 (-0.0..+0.2) |
| neighbours_s2 | neighbour_glyphs | 2 | 3.3% | +1.7 (+1.4..+2.1) * | 1.3% | +0.7 (+0.5..+0.9) * |
| neighbours_s3 | neighbour_glyphs | 3 | 32.3% | +30.7 (+27.6..+34.5) * | 9.1% | +8.6 (+7.6..+9.6) * |
| grid_s1 | grid_paper | 1 | 1.7% | +0.1 (-0.1..+0.2) | 0.6% | -0.0 (-0.1..+0.1) |
| grid_s2 | grid_paper | 2 | 1.7% | +0.1 (-0.0..+0.3) | 0.6% | +0.0 (-0.1..+0.1) |
| grid_s3 | grid_paper | 3 | 1.9% | +0.3 (+0.1..+0.5) * | 0.7% | +0.1 (-0.0..+0.2) |
| ruled_s3 | grid_paper | 3 | 1.6% | -0.0 (-0.1..+0.0) | 0.6% | -0.0 (-0.1..+0.0) |
| erosion_s1 | morphology | 1 | 1.7% | +0.1 (-0.0..+0.1) | 0.5% | -0.1 (-0.2..+0.0) |
| erosion_s2 | morphology | 2 | 1.7% | +0.1 (-0.0..+0.2) | 0.6% | -0.0 (-0.1..+0.1) |
| erosion_s3 | morphology | 3 | 3.2% | +1.6 (+1.2..+2.0) * | 1.6% | +1.0 (+0.8..+1.2) * |
| dilation_s1 | morphology | 1 | 1.9% | +0.2 (+0.1..+0.4) * | 0.7% | +0.1 (-0.0..+0.2) |
| dilation_s3 | morphology | 3 | 3.9% | +2.3 (+1.9..+2.7) * | 1.7% | +1.1 (+0.9..+1.4) * |
| elastic_s1 | elastic | 1 | 1.7% | +0.0 (-0.0..+0.1) | 0.6% | +0.1 (-0.0..+0.2) |
| elastic_s2 | elastic | 2 | 1.6% | +0.0 (-0.1..+0.1) | 0.6% | +0.1 (-0.0..+0.2) |
| elastic_s3 | elastic | 3 | 1.9% | +0.3 (+0.1..+0.4) * | 0.7% | +0.1 (-0.0..+0.2) |
| griddist_s1 | elastic | 1 | 1.8% | +0.2 (+0.0..+0.4) * | 0.6% | +0.0 (-0.1..+0.2) |
| griddist_s2 | elastic | 2 | 1.8% | +0.2 (+0.0..+0.4) | 0.7% | +0.1 (-0.0..+0.2) |
| griddist_s3 | elastic | 3 | 2.0% | +0.3 (+0.1..+0.5) * | 0.8% | +0.2 (+0.0..+0.3) * |
| phone_s1 | phone_photo | 1 | 1.6% | +0.0 (-0.1..+0.2) | 0.6% | -0.0 (-0.1..+0.1) |
| phone_s2 | phone_photo | 2 | 2.0% | +0.4 (+0.2..+0.6) * | 0.8% | +0.2 (+0.0..+0.4) * |
| phone_s3 | phone_photo | 3 | 6.9% | +5.3 (+4.6..+6.0) * | 4.3% | +3.7 (+3.1..+4.4) * |
| phone_shadow | phone_photo | 3 | 1.6% | -0.0 (-0.1..+0.1) | 0.5% | -0.0 (-0.1..+0.0) |
| phone_noise | phone_photo | 3 | 1.6% | +0.0 (-0.1..+0.1) | 0.6% | -0.0 (-0.1..+0.0) |
| phone_blur | phone_photo | 3 | 2.0% | +0.4 (+0.3..+0.6) * | 0.7% | +0.2 (+0.0..+0.3) * |
| phone_motion | phone_photo | 3 | 1.9% | +0.3 (+0.2..+0.5) * | 0.8% | +0.2 (+0.1..+0.3) * |
| phone_downscale | phone_photo | 3 | 2.4% | +0.8 (+0.6..+1.0) * | 1.0% | +0.4 (+0.2..+0.6) * |
| phone_jpeg | phone_photo | 3 | 1.6% | -0.0 (-0.1..+0.1) | 0.6% | -0.0 (-0.1..+0.1) |
| scan_clean_color | scanner | 1 | 1.6% | +0.0 (-0.1..+0.2) | 0.5% | -0.0 (-0.2..+0.1) |
| scan_grayscale | scanner | 2 | 1.6% | -0.1 (-0.2..+0.1) | 0.5% | -0.0 (-0.1..+0.1) |
| scan_photocopy | scanner | 3 | 1.9% | +0.3 (+0.1..+0.4) * | 0.7% | +0.2 (+0.0..+0.3) * |

`*` przedzial nie obejmuje zera, czyli warunek zmienia CER istotnie.
Linii bazowych: v1 1500, v2 1500

## Ranking rodzin (najgorszy warunek w rodzinie)

| rodzina | najgorszy warunek | zmiana CER (v2) |
|---|---|---|
| neighbour_glyphs | neighbours_s3 | +8.6 pkt |
| phone_photo | phone_s3 | +3.7 pkt |
| morphology | dilation_s3 | +1.1 pkt |
| elastic | griddist_s3 | +0.2 pkt |
| scanner | scan_photocopy | +0.2 pkt |
| grid_paper | grid_s3 | +0.1 pkt |

