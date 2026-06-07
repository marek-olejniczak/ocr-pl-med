Modele do benchmarku segmentacji/detekcji linii pisma odręcznego. Cztery rodziny podejść - porównujemy filozofie, nie tylko konkretne sieci. Każdy model: tryb pretrained (zero-shot) tylko jeśli był trenowany na zadaniu tekstowym, finetuning na naszym datasecie jeśli wspierany.

#przydatne

## 1. Generyczna detekcja obiektów (linia = obiekt)

### YOLOv8 / YOLO11 (Ultralytics)
- single-stage, anchor-free CNN; YOLOv8 już sprawdzony na naszym datasecie (dobre wyniki), YOLO11 jako nowsza iteracja tej samej rodziny
- tryb: tylko finetuning (pretrained na COCO - 80 klas obiektów, brak klasy "linia tekstu", zero-shot bez sensu)
- **uwaga: brak oficjalnych paperów** - Ultralytics nie publikuje artykułów dla v8/v11; cytujemy software zgodnie z zaleceniem autorów:
	- Jocher G., Chaurasia A., Qiu J., *Ultralytics YOLOv8* (2023), https://github.com/ultralytics/ultralytics (AGPL-3.0)
	- zalecenie cytowania: https://github.com/orgs/ultralytics/discussions/20178
- tło architektoniczne rodziny YOLO (oryginalny paper): Redmon J. et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016, https://arxiv.org/abs/1506.02640

### RT-DETR
- detekcja end-to-end na transformerze (rodzina DETR), bez NMS - potencjalnie ciekawe przy gęsto upakowanych liniach, gdzie NMS może zjadać sąsiednie boxy; oś porównania CNN vs transformer
- tryb: tylko finetuning (pretrained COCO); dostępny w ultralytics - ten sam format danych co YOLO
- Zhao Y. et al., *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024, https://arxiv.org/abs/2304.08069
- tło: Carion N. et al., *End-to-End Object Detection with Transformers* (DETR), ECCV 2020, https://arxiv.org/abs/2005.12872

### Mask R-CNN (Detectron2)
- klasyczny two-stage baseline akademicki; instance segmentation, ale u nas używany na bboxach
- tryb: tylko finetuning (pretrained COCO); nasz dataset jest w formacie COCO → zero konwersji
- He K., Gkioxari G., Dollár P., Girshick R., *Mask R-CNN*, ICCV 2017, https://arxiv.org/abs/1703.06870
- tło (dwustopniowa detekcja): Ren S. et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015, https://arxiv.org/abs/1506.01497

## 2. Scene-text detection (segmentacja piksel-po-pikselu + postprocessing)

### DBNet / DBNet++ (przez docTR)
- segmentacyjna detekcja tekstu z różniczkowalną binaryzacją; dominująca rodzina w text detection
- pretrained wykrywa **słowa**, nie linie → zero-shot na liniach wypadnie słabo, ale finetuning przestawia granulację - gotowy wniosek do pracy: granularność detekcji jest wyuczona, nie architekturalna
- tryb: zero-shot + finetuning
- Liao M. et al., *Real-time Scene Text Detection with Differentiable Binarization*, AAAI 2020, https://arxiv.org/abs/1911.08947
- DBNet++: Liao M. et al., *Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion*, TPAMI 2022, https://arxiv.org/abs/2202.10304
- implementacja: docTR (Mindee), https://github.com/mindee/doctr - biblioteka, bez papera

## 3. Segmentacja linii specyficzna dla HTR

### Kraken BLLA
- standard w świecie HTR (eScriptorium, dokumenty historyczne); klasyfikacja pikseli → baseline'y + poligony linii, architektura CNN+LSTM
- jedyny kandydat budowany stricte pod segmentację linii rękopisów - najbliższy domenowo
- tryb: zero-shot (model blla) + finetuning przez `ketos segtrain`; GT wymaga PAGE XML z baseline'ami - generujemy syntetyczne baseline'y z bboxów (aproksymacja, do opisania w ograniczeniach pracy)
- Kiessling B., *A Modular Region and Text Line Layout Analysis System*, ICFHR 2020, s. 313–318, https://ieeexplore.ieee.org/document/9257770 (open access: https://hal.science/hal-04442992)
- software: https://github.com/mittagessen/kraken; nowsze: Kiessling B., *Version 5 of the Kraken ATR Engine for the Humanities*, ICDAR 2025, https://dl.acm.org/doi/10.1007/978-3-032-04624-6_26

## 4. Nowoczesne modele layoutu dokumentów

### Surya (det)
- detekcja tekstu **natywnie na poziomie linii** (zmodyfikowany EfficientViT segformer) - dokładnie nasze zadanie; mierzy "ile daje współczesny model dokumentowy bez adaptacji"
- tryb: **tylko zero-shot** - finetuning detekcji niedostępny publicznie (oferowany komercyjnie przez Datalab)
- **bez papera** - software: https://github.com/datalab-to/surya (GPL-3.0, wagi z ograniczeniami komercyjnymi - dla pracy akademickiej OK)

## Dane pomocnicze

- IAM (część obecnego datasetu dev): Marti U.-V., Bunke H., *The IAM-database: an English sentence database for offline handwriting recognition*, IJDAR 5, 39–46 (2002), https://doi.org/10.1007/s100320200071

## Macierz eksperymentów (skrót)

| Model | Zero-shot | Finetuning |
|---|---|---|
| YOLOv8, YOLO11, RT-DETR, Mask R-CNN | - | tak |
| DBNet, Kraken BLLA | tak | tak |
| Surya | tak | - |

Każdy finetuning w dwóch wariantach (raw / preprocessed) × ewaluacja na obu wariantach - ablacja 2×2 preprocessingu.

## Odrzucone / odłożone

- ARU-Net (Grüning et al., https://arxiv.org/abs/1802.03345), dhSegment - TensorFlow 1.x, koszt uruchomienia nieproporcjonalny do wartości
- Doc-UFCN (Teklia) - #do_sprawdzenia, ewentualnie do dołożenia później
- PaddleOCR det - ta sama rodzina co DBNet (duplikacja), ekosystem Paddle uciążliwy
- SAM/Hi-SAM - research-grade, ciężki finetuning, poza zakresem pracy inż.
