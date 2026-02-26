
# Prace pokrewne

## Scalable handwritten text recognition system for lexicographic sources of under-resourced languages and alphabets
https://iccs-meeting.org/archive/iccs2021/papers/127420135.pdf
### Skalowalny system HTR dla polskojęzycznych źródeł leksykograficznych

**1. Cel i zakres prac** Celem opisanego projektu było opracowanie i wdrożenie zautomatyzowanego systemu rozpoznawania tekstu rękopiśmiennego (HTR) do dygitalizacji rozbudowanej kartoteki „Słownika języka polskiego XVII i XVIII wieku”, zawierającej 2,8 miliona odręcznie zapisanych kart. System zaprojektowano z myślą o skalowalności oraz obsłudze specyficznych cech języka polskiego, takich jak znaki diakrytyczne, w warunkach ograniczonej dostępności ręcznie etykietowanych danych treningowych.

**2. Architektura systemu i metodologia** Proces przetwarzania danych podzielono na trzy główne etapy: detekcję, rozpoznawanie oraz postprocessing.

- **Detekcja hasła (Detection):** Do lokalizacji słów nagłówkowych na kartach wykorzystano model **Keras OCR CRAFT**. W celu optymalizacji wydajności obliczeniowej analizę ograniczono do górnego fragmentu obrazu (300 pikseli), gdzie statystycznie najczęściej znajdowało się hasło.

- **Rozpoznawanie (Recognition):** Zastosowano hybrydową architekturę głębokiego uczenia **TPS-ResNet-BiLSTM-CTC**:

    - **Spatial Transformer Network (STN):** Wykorzystuje transformację _Thin Plate Spline_ (TPS) do rektyfikacji obrazu, co pozwala na niwelowanie zniekształceń geometrycznych i zmienności stylu pisma.

    - **Ekstrakcja cech (CNN):** Zastosowano sieć **ResNet** jako szkielet (backbone) do wyodrębniania cech wizualnych charakterystycznych dla poszczególnych liter.

    - **Warstwa rekurencyjna (BiLSTM):** Sekwencyjne przetwarzanie cech w celu uchwycenia zależności kontekstowych między znakami.

    - **Warstwa CTC (Connectionist Temporal Classification):** Umożliwia mapowanie predykcji na sekwencje znaków bez konieczności uprzedniej segmentacji obrazu na poszczególne litery, co jest kluczowe w przypadku pisma cursive.

- **Postprocessing:** Wdrożono algorytm **Constrained Word Beam Search (WBS)**. Wyniki modelu są dopasowywane do zamkniętego słownika 86 000 haseł. Dodatkowo wykorzystano informację o alfabetycznym uporządkowaniu kartotek (podział na szuflady), co pozwoliło na drastyczne ograniczenie przestrzeni poszukiwań i poprawę trafności predykcji.


**3. Zbiory danych i proces uczenia** Ze względu na brak wystarczającej liczby naturalnych przykładów polskiego pisma ręcznego, zastosowano podejście oparte na **transfer learningu** i danych syntetycznych.

- **Dane syntetyczne:** Wygenerowano zbiór **PL-500k-synthetic**, składający się z 500 000 obrazów polskich słów utworzonych przy użyciu różnorodnych fontów i technik augmentacji (m.in. posteryzacja, wyrównywanie histogramu, transformacje afiniczne).

- **Obsługa diakrytyków:** Przygotowano dedykowany zestaw **PL-30k-diacritics** zawierający losowe ciągi z równomiernym rozkładem polskich znaków diakrytycznych, co pozwoliło modelowi lepiej rozpoznawać rzadziej występujące litery (np. „ą”, „ę”, „ć”).

- **Transfer Learning:** Model wstępnie wytrenowany na zbiorach angielskojęzycznych (MJSynth, SynthText, CVIT – 9 mln słów) był sukcesywnie dotrenowywany na polskich danych syntetycznych.

- **Zbiór walidacyjny:** Na potrzeby ewaluacji przygotowano ręcznie etykietowany zbiór 20 000 obrazów kart (PL-20k-hand-labelled).


**4. Wyniki i wnioski**

- **Skuteczność detekcji:** Moduł detekcji osiągnął współczynnik IoU (Intersection over Union) na poziomie **0,93**.

- **Skuteczność rozpoznawania:** Najwyższą dokładność na poziomie słowa (**word-level accuracy**) uzyskano przy użyciu ograniczonego algorytmu WBS, osiągając wynik **0,881 (88,1%)**.

- **Wnioski techniczne:** Wykazano, że modele typu _best path decoding_ (bez zewnętrznych słowników) osiągają znacznie niższą dokładność (ok. 37-43%), co potwierdza kluczową rolę postprocessingu opartego na słowniku dziedzinowym w zadaniach OCR pisma ręcznego. Stwierdzono również, że popularny zbiór danych IAM nie jest optymalny do trenowania modeli pracujących na rzeczywistych, „brudnych” skanach archiwalnych ze względu na nadmierne dopasowanie (overfitting) do idealnie przyciętych próbek.


## Interpreting Doctors' Notes: Handwriting Recognition & Deep Learning
https://www.ijcaonline.org/archives/volume187/number45/talekar-2025-ijca-925199.pdf
### Hybrydowy system rozpoznawania rękopisów medycznych

**1. Cel i przedmiot opracowania** Przedmiotem prac jest hybrydowy system oparty na sztucznej inteligencji, dedykowany do konwersji odręcznych recept lekarskich na tekst cyfrowy. Rozwiązanie to ma na celu minimalizację błędów medycznych wynikających z nieczytelnego pisma oraz wsparcie dygitalizacji dokumentacji zdrowotnej.

**2. Metodologia i architektura systemu** System wykorzystuje wieloetapowy proces przetwarzania, łączący tradycyjne narzędzia OCR z zaawansowanymi modelami uczenia głębokiego:

- **Wstępne przetwarzanie obrazu (Preprocessing):** Wykorzystano bibliotekę **OpenCV** do poprawy jakości wejściowych skanów recept. Kluczowe operacje obejmują:

    - Konwersję do skali szarości w celu redukcji złożoności danych.

    - Redukcję szumów przy użyciu rozmycia Gaussa i operacji morfologicznych.

    - Normalizację obrazów do stałego rozmiaru 128x128 pikseli.

    - Binaryzację (progowanie) dla zwiększenia kontrastu między tekstem a tłem.

- **Hybrydowa architektura modelu:**

    - **Tesseract OCR:** Wykorzystywany w pierwszej fazie do detekcji układu tekstu i generowania wstępnych wyników rozpoznawania.

    - **CNN (Convolutional Neural Networks):** Pełni rolę ekstraktora cech, identyfikując wzorce wizualne, takie jak pociągnięcia pióra, krzywizny i kształty znaków.

    - **LSTM (Long Short-Term Memory):** Warstwa rekurencyjna odpowiedzialna za modelowanie zależności sekwencyjnych w tekście odręcznym, co umożliwia interpretację ciągów znaków bez ich jawnej segmentacji.

    - **Funkcja straty CTC (Connectionist Temporal Classification):** Zastosowana do dopasowania przewidywanych sekwencji do prawdy terenowej (_ground truth_) w przypadku danych nieosegmentowanych.


**3. Implementacja i proces uczenia**

- **Narzędzia:** Model zaimplementowano w językach Python z wykorzystaniem bibliotek **TensorFlow** oraz **Keras**. Obliczenia przyspieszono dzięki akceleracji GPU.

- **Dane treningowe:** Zbiór danych obejmował obrazy recept pochodzące z placówek medycznych oraz publicznych baz danych, takich jak IAM-OnDB. Dane poddano augmentacji (rotacja, skalowanie, przesunięcie) w celu zwiększenia odporności modelu na zmienność pisma.

- **Parametry treningu:** Zastosowano optymalizator **Adam** (learning rate: 0.001) oraz technikę _early stopping_ dla zapobiegania przeuczeniu.


**4. Wyniki i wskaźniki wydajności** System został oceniony pod kątem dokładności i szybkości działania:

- **Dokładność na poziomie znaków:** Uzyskano wynik przekraczający **91,3%**.

- **Współczynnik błędów (CER):** Utrzymano błąd poniżej **8%**.

- **Czas przetwarzania:** Średni czas analizy pojedynczego obrazu wynosi od **1,8 do 2 sekund**, co pozwala na wdrożenie systemu w trybie czasu rzeczywistego.


**5. Ograniczenia i wnioski** Zauważono, że skuteczność rozpoznawania spada w przypadku ekstremalnie nieczytelnego pisma oraz niskiej jakości obrazów (np. cieni lub niskiej rozdzielczości). Wskazano na konieczność dalszego rozwoju systemu w kierunku obsługi wielu języków oraz integracji z bazami danych leków w celu automatycznej weryfikacji dawek i przeciwwskazań.


## MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning
https://arxiv.org/pdf/2505.20513v1
### Personalizacja systemów HTR poprzez Meta-Learned Prompt Tuning

**1. Cel i problematyka** Praca adresuje kluczowe ograniczenie współczesnych modeli HTR: trudność w generalizacji na specyficzne, unikalne style pisma bez konieczności kosztownego douczania całego modelu (fine-tuning) na danych etykietowanych dla każdego nowego autora. W kontekście medycznym odpowiada to problemowi adaptacji systemu do skrajnie różnych i nieczytelnych charakterów pisma poszczególnych lekarzy przy braku dostępu do etykiet tekstowych w czasie rzeczywistym (test-time).

**2. Architektura i innowacja: Framework MetaWriter** Zaproponowane podejście odchodzi od klasycznego douczania wag modelu na rzecz **Prompt Tuning** osadzonego w paradygmacie **Meta-Learningu**.

- **Personalizacja jako Prompt Tuning:** Zamiast modyfikacji wag _backbone'u_ (np. warstw splotowych czy bloków Transformer), model optymalizuje jedynie niewielki zestaw wyuczalnych tokenów (promptów) wstrzykiwanych do architektury. Pozwala to na aktualizację **poniżej 1% całkowitej liczby parametrów** modelu.

- **Adaptacja bez nadzoru (Self-supervised Test-Time Adaptation):** Innowacja polega na wykorzystaniu pomocniczego zadania **rekonstrukcji obrazu** (auxiliary image reconstruction task) jako sygnału do adaptacji w czasie testu. Model dostraja prompty tak, aby zminimalizować stratę rekonstrukcji na nieetykietowanych próbkach od danego autora, co pośrednio wymusza na modelu uchwycenie cech specyficznych dla jego stylu pisma.

- **Meta-inicjalizacja (Meta-Learning):** Aby proces adaptacji na kilku próbkach był skuteczny, parametry promptów nie są inicjalizowane losowo. Wykorzystano meta-learning (prawdopodobnie zbliżony do MAML lub Reptile), aby znaleźć **optymalną inicjalizację promptów**, która pozwala na szybką zbieżność i minimalizację błędu rozpoznawania tekstu po wykonaniu zaledwie kilku kroków gradientowych na zadaniu pomocniczym.


**3. Aspekty techniczne i optymalizacja**

- **Efektywność obliczeniowa:** Dzięki ograniczeniu adaptacji do promptów, narzut pamięciowy i obliczeniowy podczas fazy testowej jest minimalny, co umożliwia wdrożenie na urządzeniach o ograniczonych zasobach (resource-constrained scenarios).

- **Stabilność:** Zastosowanie self-supervised loss zapobiega katastrofalnemu zapominaniu (catastrophic forgetting) oraz dryfowi parametrów, który często występuje przy niekontrolowanym fine-tuningu na małych zbiorach.

- **Redukcja parametrów:** Metoda osiąga wyniki wyższe niż dotychczasowe State-of-the-Art (SotA) przy użyciu **20-krotnie mniejszej liczby modyfikowalnych parametrów**.


**4. Wyniki i walidacja**

- **Zbiory danych:** Model był walidowany na standardowych benchmarkach **IAM Handwriting Database** oraz **RIMES**.

- **Wydajność:** System konsekwentnie przewyższa poprzednie metody meta-learningowe oparte na gradientach, eliminując jednocześnie potrzebę posiadania etykiet w czasie adaptacji.


**5. Wnioski pod kątem implementacji w OCR medycznym** Dla projektu OCR dokumentacji medycznej praca ta sugeruje następujące kroki techniczne:

1. Zastosowanie architektury typu Transformer jako bazy (np. TrOCR lub ViT+seq2seq).

2. Wprowadzenie mechanizmu **Test-Time Adaptation**: gdy system otrzymuje serię skanów od jednego lekarza, może "nauczyć się" jego stylu poprzez rekonstrukcję obrazu (np. przez autoenkoder lub warstwy dekodujące obraz), co automatycznie poprawi precyzję dekodowania tekstu bez ręcznej korekty haseł.

3. Wykorzystanie meta-learningu na etapie pre-trainingu, aby przygotować uniwersalne prompty zdolne do błyskawicznej adaptacji do nowych, specyficznych krojów pisma.

## AI-Based OCR System for Handwritten Medical Prescription Recognition and Interpretation
https://www.indjcst.com/archiver/archives/ai_based_ocr_system_for_handwritten_medical_prescription_recognition_and_interpretation.pdf

#przydatne
### System OCR oparty na AI do rozpoznawania i interpretacji odręcznych recept medycznych

1. **Cel i przedmiot opracowania:** Przedmiotem prac jest system OCR oparty na sztucznej inteligencji, przeznaczony do rozpoznawania i interpretacji odręcznych recept medycznych. Rozwiązanie to ma na celu minimalizację błędów wynikających z niskiej czytelności i braku standaryzacji, co prowadzi m.in. do błędnego dawkowania, niepożądanych reakcji na leki i opóźnień w leczeniu. Projekt wspiera cyfrową transformację opieki zdrowotnej, redukując błędy ludzkie i poprawiając bezpieczeństwo pacjentów.

2. **Metodologia i architektura systemu:** System wykorzystuje wieloetapowy przepływ pracy, integrujący tradycyjne narzędzia OCR z modelami głębokiego uczenia i przetwarzaniem języka naturalnego (NLP).

	- **Wstępne przetwarzanie obrazu (Preprocessing):** Etap ten znacząco poprawia jakość obrazów wejściowych przed właściwą analizą. Kluczowe operacje obejmują:

	    - Binaryzację w celu wyraźniejszego wyodrębnienia tekstu z tła.

	    - Usuwanie szumów, takich jak smugi, artefakty i nieistotne oznaczenia.

	    - Korektę pochylenia (skew correction) dla wyrównania skanów.

	    - Segmentację wierszy tekstu w celu uproszczenia procesu rozpoznawania znaków.

	    - Normalizację standaryzującą rozmiar i rozdzielczość obrazów dla spójnej pracy modelu.

	- **Hybrydowa architektura modelu i ekstrakcja danych:**

	    - **Tesseract OCR i CNN-LSTM:** Tradycyjne narzędzie Tesseract zostało zintegrowane z modelami CNN-LSTM, co zapewnia niezawodne rozpoznawanie odręcznych znaków i symboli pomimo niespójnych stylów pisma.

	    - **Moduł NLP i NER:** Wykorzystuje zaawansowane techniki rozpoznawania do ekstrakcji i klasyfikacji kluczowych informacji medycznych, takich jak nazwy leków, dawkowanie, dane pacjenta i instrukcje administracyjne.


 3. **Implementacja i proces uczenia**

- **Narzędzia i środowisko:** System zaimplementowano w języku Python 3.x, wykorzystując frameworki TensorFlow/Keras do modelowania architektur CNN-LSTM. Do interpretacji NLP użyto bibliotek takich jak SpaCy oraz BERT, a intuicyjny interfejs użytkownika stworzono za pomocą platformy Streamlit. Ustrukturyzowane wyniki są bezpiecznie przechowywane w bazach danych SQLite lub PostgreSQL.

- **Dane treningowe:** Wykorzystano rzeczywiste zbiory danych odręcznych recept, które odzwierciedlają różne style pisma, specyficzne dla lekarzy skróty oraz treści wielojęzyczne. Dane te były adnotowane ręcznie przez ekspertów, aby dokładnie oznaczyć kluczowe encje medyczne.

3. **Wyniki i wskaźniki wydajności** Prototyp systemu został poddany ocenie pod kątem dokładności i szybkości, wskazując na dużą przydatność w warunkach klinicznych i aptecznych:

	- **Dokładność rozpoznawania (Recognition Accuracy):** System osiągnął wysoką dokładność na poziomie 92% dla ustrukturyzowanych recept.

	- **Precyzja i czułość:** Algorytmy poprawnie wyodrębniały terminy dotyczące leków i dawkowania, osiągając precyzję na poziomie 90%. Czułość (Recall) wyniosła 88%, przy czym odnotowano pewne trudności z rzadko spotykanymi terminami skrótowymi.

	- **Czas przetwarzania (Average Latency):** Średni czas analizy wynosi 1,5 sekundy na obraz, co czyni system w pełni dostosowanym do działania w czasie rzeczywistym.

	- **Satysfakcja użytkowników:** Aż 91% użytkowników oceniło system jako intuicyjny i bardzo pomocny w codziennej pracy.


4. **Ograniczenia i wnioski**
	Zauważono, że system jest obecnie ograniczony wielkością zbioru danych treningowych i wymaga rozszerzenia o bardziej zróżnicowane próbki z wielu instytucji. Istotnym wyzwaniem pozostaje szeroka zmienność skrótów używanych przez lekarzy oraz zależność od jakości obrazu, gdzie np. zamazane skany obniżają dokładność rozpoznawania. Prototyp skupia się głównie na języku angielskim i nie posiada jeszcze aktywnej integracji z systemami szpitalnymi EHR w czasie rzeczywistym. Wskazano na konieczność dalszego rozwoju poprzez wdrożenie obsługi wielojęzycznej, zastosowanie bezpiecznych API do łączności z systemami zarządzania apteką oraz wykorzystanie specjalistycznych modeli transformatorowych, takich jak BioBERT i MedBERT.

## Advancements and Challenges in Handwritten Text Recognition: A Comprehensive Survey
https://pmc.ncbi.nlm.nih.gov/articles/PMC10817575/pdf/jimaging-10-00018.pdf

#do_sprawdzenia
Porównanie wielu technik


# Pozostałe pokrewne

## Architektura Detekcji i Rozpoznawania (Model Hybrydowy)

Głównym wyzwaniem jest stworzenie modelu na tyle lekkiego, by mógł działać lokalnie (edge deployment) bez przesyłania danych pacjentów na zewnętrzne serwery.

- **Praca:** **Joan Puigcerver (ICDAR 2017)**
    - **URL:** [https://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf](https://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf)
    - **Kluczowe wnioski:** Wielowymiarowe sieci MDLSTM są zbyt zasobożerne i trudne do optymalizacji. Zamiast nich rekomenduje się architekturę hybrydową: głęboką sieć **CNN + 1D-LSTM**.
    - **Szczegóły techniczne:** Model wykorzystuje 5 bloków konwolucyjnych z małymi jądrami ($3\times3$ px), co pozwala zachować detale polskiej diakrytyki (kropki, ogonki). Całość wieńczy funkcja straty **CTC (Connectionist Temporal Classification)**, która eliminuje konieczność ręcznego dzielenia wyrazów na litery


## Inżynieria Syntezy Danych (Generowanie Recept)

Z powodu ograniczeń RODO w dostępie do autentycznych recept , system musi opierać się na danych syntetycznych tworzonych z banków znaków takich jak PHCD.

- **Praca: ScrabbleGAN**
    - **URL:** [https://arxiv.org/pdf/2005.13044](https://arxiv.org/pdf/2005.13044)
    - **Kluczowe wnioski:** Rozwiązuje problem generowania wyrazów o dowolnej długości (np. długich nazw chemicznych leków) poprzez "zlepianie" filtrów znaków w osi szerokości. Posiada wbudowany **Recognizer**, który pełni rolę "nauczyciela", karząc generator za nieczytelne napisy.
        
- **Praca: GANwriting**
    - **URL:** [https://arxiv.org/pdf/2003.02567](https://arxiv.org/pdf/2003.02567)
    - **Kluczowe wnioski:** Skupia się na naśladowaniu indywidualnych stylów pisma (n-shot) poprzez rozdzielenie kodera stylu od kodera treści. Słabość – traci jakość przy wyrazach dłuższych niż 10 znaków.
        
- **Praca: Handwriting Transformers (HWT)**
    - **URL:** [https://arxiv.org/pdf/2102.08742](https://arxiv.org/pdf/2102.08742)
    - **Kluczowe wnioski:** Uznana za rozwiązanie **State-of-the-Art**. Dzięki mechanizmom **Self-Attention** i **Cross-Attention** potrafi skorelować kształty liter na początku i końcu długiego zdania. Jest to kluczowe dla polskiej nomenklatury medycznej i wyrazów spoza słownika (OOV).
        
- **Praca: Learning to Read and Write (InkSight)**
    - **URL:** [https://arxiv.org/pdf/2009.00678](https://arxiv.org/pdf/2009.00678)
    - **Kluczowe wnioski:** Praca ta dostarcza mechanizmów dyskryminatorów częstotliwościowych, które pomagają usuwać mikroskopijne artefakty graficzne, zapewniając fotorealizm syntetycznych dokumentów.


## Adaptacja do Autora (Przełamywanie luki generalizacyjnej)

Modele często tracą celność, gdy napotykają pismo nowego lekarza (spadek o 11-13%). 

- **Praca: Sieci SEN (Style Extractor Network)**
    - **URL:** [https://www.researchgate.net/publication/356903770_Fast_writer_adaptation_with_style_extractor_network_for_handwritten_text_recognition](https://www.researchgate.net/publication/356903770_Fast_writer_adaptation_with_style_extractor_network_for_handwritten_text_recognition)
    - **Kluczowe wnioski:** To podejście "lekkie" i rekomendowane dla szpitali. Tworzy matematyczny "odcisk palca" grafomotoryki lekarza (wektor stylu) i wstrzykuje go do zamrożonego modelu głównego. Nie wymaga czasochłonnego douczania całej sieci.
        
- **Praca: MetaHTR (MAML)**
    - **URL:** [https://arxiv.org/pdf/2307.15071](https://arxiv.org/pdf/2307.15071)
    - **Kluczowe wnioski:** Wykorzystuje meta-uczenie (uczenie się jak się uczyć), by przygotować model do błyskawicznej adaptacji na podstawie zaledwie 16-20 próbek pisma. Choć bardzo skuteczne w redukcji błędu WER, jest kosztowne obliczeniowo (pochodne drugiego rzędu)