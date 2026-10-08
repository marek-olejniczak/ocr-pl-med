## CER ogolem (95% bootstrap po liniach)

| model | n | CER | 95% CI | exact |
|---|---|---|---|---|
| v2_200k | 2160 | 12.7% | 12.2% - 13.2% | 26.2% |
| bez_short_words | 2160 | 12.6% | 12.1% - 13.1% | 26.2% |
| bez_caps | 2160 | 14.5% | 13.8% - 15.2% | 23.9% |
| bez_arrows_bullets | 2160 | 12.8% | 12.3% - 13.3% | 25.5% |
| bez_anatomy_vocab | 2160 | 14.7% | 14.1% - 15.2% | 21.1% |
| bez_neighbour_glyphs | 2160 | 13.2% | 12.6% - 13.8% | 24.7% |
| bez_grid_paper | 2160 | 12.9% | 12.4% - 13.4% | 25.6% |
| bez_morphology | 2160 | 12.8% | 12.3% - 13.3% | 25.8% |
| bez_elastic | 2160 | 12.8% | 12.3% - 13.3% | 25.8% |
| bez_phone_photo | 2160 | 12.8% | 12.3% - 13.3% | 25.8% |
## Zmiana wzgledem v2_200k (pary linia po linii)

Dodatnia = gorzej niz punkt odniesienia. `*` = 95% przedzial nie obejmuje zera.

| model | CER | zmiana | 95% CI zmiany | linii lepiej / gorzej |
|---|---|---|---|---|
| bez_anatomy_vocab | 14.7% | +1.93 pkt * | +1.57 .. +2.32 | 291 / 643 |
| bez_caps | 14.5% | +1.75 pkt * | +1.25 .. +2.27 | 285 / 310 |
| bez_neighbour_glyphs | 13.2% | +0.47 pkt * | +0.22 .. +0.76 | 294 / 361 |
| bez_grid_paper | 12.9% | +0.16 pkt | -0.00 .. +0.34 | 221 / 232 |
| bez_morphology | 12.8% | +0.12 pkt | -0.05 .. +0.28 | 220 / 233 |
| bez_arrows_bullets | 12.8% | +0.07 pkt | -0.12 .. +0.26 | 245 / 269 |
| bez_elastic | 12.8% | +0.06 pkt | -0.10 .. +0.22 | 214 / 204 |
| bez_phone_photo | 12.8% | +0.06 pkt | -0.12 .. +0.24 | 247 / 249 |
| bez_short_words | 12.6% | -0.13 pkt | -0.33 .. +0.07 | 285 / 240 |


## CER wg zrodla (autora)

| zrodlo | n | v2_200k | bez_short_words | bez_caps | bez_arrows_bullets | bez_anatomy_vocab | bez_neighbour_glyphs | bez_grid_paper | bez_morphology | bez_elastic | bez_phone_photo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13) Unaczynienie Miednica | 187 | 4.1% | 4.5% | 6.5% | 4.8% | 5.6% | 4.7% | 4.7% | 4.4% | 4.3% | 4.5% |
| Elementy Topograficzne | 80 | 10.0% | 10.0% | 11.7% | 9.6% | 12.0% | 10.0% | 10.0% | 10.3% | 10.4% | 10.0% |
| Noga | 404 | 11.5% | 11.5% | 13.6% | 11.6% | 13.5% | 11.8% | 11.6% | 11.3% | 11.3% | 11.6% |
| Serce - Więcej Niż Lek | 263 | 8.6% | 8.6% | 9.4% | 8.4% | 9.6% | 9.0% | 8.9% | 8.5% | 9.1% | 8.9% |
| Zrzut 05-07 | 356 | 13.6% | 13.1% | 14.2% | 13.6% | 15.2% | 14.3% | 13.7% | 13.8% | 13.6% | 13.6% |
| Zrzut 05-12 | 394 | 13.0% | 12.6% | 15.2% | 13.2% | 14.9% | 13.3% | 13.2% | 13.2% | 13.0% | 12.9% |
| anatomia_zmysly | 357 | 23.8% | 24.0% | 28.7% | 23.8% | 27.9% | 25.1% | 23.6% | 24.0% | 23.8% | 23.8% |
| data | 119 | 22.6% | 22.3% | 22.1% | 22.4% | 25.4% | 22.3% | 22.6% | 22.6% | 22.2% | 22.2% |

## CER wg dlugosci GT

| znaki | n | v2_200k | bez_short_words | bez_caps | bez_arrows_bullets | bez_anatomy_vocab | bez_neighbour_glyphs | bez_grid_paper | bez_morphology | bez_elastic | bez_phone_photo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-5 | 199 | 25.2% | 27.3% | 25.9% | 23.7% | 30.3% | 27.8% | 25.0% | 24.7% | 24.6% | 24.0% |
| 6-12 | 634 | 16.1% | 16.1% | 19.4% | 16.2% | 19.6% | 16.9% | 16.2% | 16.1% | 16.1% | 16.2% |
| 13-25 | 773 | 12.2% | 12.1% | 15.6% | 12.5% | 14.7% | 12.6% | 12.5% | 12.5% | 12.3% | 12.5% |
| 26-45 | 399 | 12.0% | 11.7% | 12.5% | 11.9% | 13.1% | 12.3% | 12.0% | 12.0% | 12.1% | 11.9% |
| 46+ | 155 | 11.3% | 10.9% | 11.4% | 11.5% | 12.2% | 11.7% | 11.5% | 11.4% | 11.2% | 11.2% |

## Najczestsze podmiany znakow

- **v2_200k**: k->h 373, a->o 265, ś->s 104, ł->t 77, t->ł 73, k->l 69, k->b 69, c->u 65
- **bez_short_words**: k->h 423, a->o 258, ś->s 117, ł->t 81, t->ł 64, r->n 63, k->b 63, c->u 62
- **bez_caps**: k->h 383, a->o 265, ś->s 118, O->o 109, c->u 65, ł->t 63, t->ł 60, k->l 60
- **bez_arrows_bullets**: k->h 404, a->o 246, ś->s 121, ł->t 102, t->ł 70, k->b 65, k->l 59, c->u 57
- **bez_anatomy_vocab**: k->h 428, a->o 255, ł->t 122, ś->s 111, k->l 82, t->ł 70, c->u 61, ł->T 60
- **bez_neighbour_glyphs**: k->h 434, a->o 251, ś->s 117, t->ł 87, c->u 75, k->l 73, ł->t 70, c->w 70
- **bez_grid_paper**: k->h 388, a->o 257, ś->s 113, ł->t 87, k->l 76, t->ł 75, k->b 64, r->n 62
- **bez_morphology**: k->h 384, a->o 263, ś->s 109, ł->t 84, k->l 79, t->ł 78, c->u 63, k->b 61
- **bez_elastic**: k->h 383, a->o 254, ś->s 115, k->l 94, t->ł 80, ł->t 76, c->u 74, r->n 57
- **bez_phone_photo**: k->h 396, a->o 271, ś->s 107, t->ł 93, ł->t 68, k->b 61, k->l 57, o->a 55
