## Text line segmentation from struck-out handwritten document images
https://www.sciencedirect.com/science/article/pii/S0957417422014075?via%3Dihub

#przydatne 

## Segmentacja linii tekstu w swobodnych dokumentach odręcznych: przypadek tekstu przekreślonego

1. **Cel i przedmiot opracowania** 
	Głównym przedmiotem artykułu jest problem segmentacji linii tekstu w tzw. "swobodnych" dokumentach odręcznych (np. notatkach, pracach egzaminacyjnych), w których często występują błędy korygowane przez autorów poprzez przekreślanie, wymazywanie lub nadpisywanie. Celem pracy jest opracowanie metody, która poradzi sobie z segmentacją zarówno czystego tekstu, jak i tekstu przekreślonego (ang. _struck-out text_), co stanowi wyzwanie dla dotychczasowych metod, które często gubią takie fragmenty lub błędnie łączą linie. Jest to kluczowe dla systemów oceniania prac, identyfikacji fałszerstw czy rozpoznawania autora.

2. **Metodologia i architektura systemu** 
	Zaproponowano trzystopniowe podejście, które jest niezależne od rodzaju pisma (skryptu) i rodzaju przekreślenia.

	- **Krok 1: Detekcja komponentów (Component Detection):** System identyfikuje komponenty na poziomie słów (zarówno czyste, jak i przekreślone) w oparciu o szacowanie szerokości pociągnięcia (Stroke Width - SW). Zastosowano tu wygładzanie pionowe i poziome oraz operacje morfologiczne, aby scalić znaki słowa w jeden obiekt.
    
	- **Krok 2: Klasyfikacja tekstu przekreślonego (Struck-out Classification):** Wykorzystano model głębokiego uczenia **DenseNet121** (zmodyfikowany), aby odróżnić komponenty przekreślone od czystych. Model został wytrenowany na stworzonym syntetycznie zbiorze danych, aby zrównoważyć liczbę przykładów przekreślonych i czystych. Klasyfikator radzi sobie z różnymi typami przekreśleń (np. pojedyncza linia, "zygzak", zamazywanie, przekreślenie krzyżykowe).
    
	- **Krok 3: Segmentacja linii tekstu (Line Segmentation):** Metoda wykorzystuje relacje przestrzenne. Kluczową innowacją jest to, że wykryte w kroku 2 słowa przekreślone są traktowane inaczej – system ignoruje je przy wyznaczaniu głównego kierunku linii, aby nie zaburzały geometrii. Analizowane cechy to:
	    - Odległość euklidesowa między punktami skrajnymi sąsiednich komponentów.
        
	    - Kąt nachylenia (slope) między środkami ciężkości (Center of Gravity - CG) komponentów.
        
	    - Wspólny obszar nakładania się (overlapping region) między sąsiadami.

3. **Wyniki i wskaźniki wydajności** 
	Zaproponowana metoda została porównana z istniejącymi rozwiązaniami i wykazała wyższą skuteczność:
	- **Skuteczność segmentacji:** Na dedykowanym zbiorze danych metoda osiągnęła wskaźnik Detection Rate (DR) na poziomie **0,94** oraz Recognition Accuracy (RA) **0,90**. Dla porównania, metody konkurencyjne osiągnęły DR na poziomie 0,74 – 0,85.
    
	- **Niezależność od skryptu:** Metoda sprawdziła się również na zbiorach wielojęzycznych (ICDAR2013) i historycznych chińskich (ICDAR2019), co potwierdza jej uniwersalność.
    
	- **Klasyfikacja przekreśleń:** Model DenseNet osiągnął F-score na poziomie **0,87** dla klasyfikacji słów przekreślonych, przewyższając metody oparte na cechach ręcznych (0,73) i prostszych sieciach CNN (0,83).
    

4. **Ograniczenia i wnioski** 
	Autorzy wskazują na pewne ograniczenia systemu:
	
	- **Szum i kontekst:** W przypadku bardzo zaszumionych znaków dochodzi czasem do błędnej klasyfikacji jako przekreślenie.
    
	- **Nieregularne odstępy:** Metoda może mieć trudności, gdy odstępy między słowami w jednej linii są bardzo zróżnicowane i nieregularne.
    
	- **Złożoność obliczeniowa:** Wykorzystanie analizy spójnych składowych (Connected Component Analysis) wymaga sporych zasobów obliczeniowych.
    

Wnioski końcowe sugerują, że uwzględnienie przetwarzania języka naturalnego (NLP) i semantyki mogłoby w przyszłości pomóc w lepszym grupowaniu słów w linie, gdy same cechy geometryczne są niewystarczające.


## LineTR: Unified Text Line Segmentation for Challenging Palm Leaf Manuscripts
### LineTR: Uniwersalna segmentacja linii tekstu dla trudnych rękopisów na liściach palmowych

#do_sprawdzenia - wydaje się bardzo przydatne, imponujace rezultaty, ale nie wiem czy medyczne dokumenty beda miały na tyle jednolitą strukturę, żeby działało to tak dobrze.

1. **Cel projektu** 
	LineTR to uniwersalny system do precyzyjnej segmentacji linii tekstu w trudnych i różnorodnych rękopisach historycznych (głównie na liściach palmowych). Rozwiązuje on problem dotychczasowych metod, które wymagały tworzenia osobnych modeli lub dostrajania parametrów pod każdy nowy zbiór danych.

2. **Jak działa LineTR?** 
	System działa w dwóch etapach i opiera się na przetwarzaniu fragmentów obrazu o adaptacyjnym, zmiennym rozmiarze (Context-Adaptive Patching):
	- **Etap 1:** Sieć typu DETR przewiduje parametryczne "bazgroły" (scribbles) przecinające linie tekstu, a hybrydowy moduł CNN-Transformer generuje ciągłą, binarną mapę energii tekstu.
	- **Etap 2:** Algorytm generowania szwów (seam generation) wykorzystuje te dane, aby obrysować każdą linię tekstu ciasno dopasowanym wielokątem (polygonem).

3. **Zbiory danych** 
	Model trenowano na połączonych, publicznych archiwach historycznych. Dodatkowo autorzy udostępnili 3 nowe, bardzo zróżnicowane zbiory rękopisów indyjskich i azjatyckich (WM, UB, SM), aby przetestować model w warunkach "zero-shot" (czyli na danych, których model nigdy wcześniej nie widział).

4. **Wyniki i wnioski**
	- **Skuteczność:** Pojedynczy model LineTR znacząco i konsekwentnie deklasuje wszystkie konkurencyjne rozwiązania (m.in. Palmira, SeamFormer) pod kątem precyzji (metryki IoU i AvgHD).
	- **Generalizacja:** Osiąga znakomite wyniki na nowych, niewidzianych wcześniej dokumentach (zero-shot), udowadniając swoją uniwersalność.
	- **Znaczenie:** LineTR eliminuje konieczność żmudnego, ręcznego dostrajania sieci do nowych kolekcji historycznych, co drastycznie ułatwia masową cyfryzację archiwów.


## Text-line extraction from handwritten document images using GAN

#przydatne 

1. **Cel projektu** 
	System rozwiązuje problem ekstrakcji linii tekstu (TLE) z trudnych, swobodnych dokumentów odręcznych. W przeciwieństwie do tradycyjnych metod opartych na regułach, które zawodzą przy nakładających się lub pochylonych liniach, zaproponowano nowatorskie podejście oparte na głębokim uczeniu. Po raz pierwszy w literaturze zastosowano do tego celu generatywne sieci przeciwstawne (GAN), traktując podział na linie jako zadanie translacji obrazu na obraz.

2. **Jak działa system?** 
	Działanie systemu polega na automatycznym generowaniu obrazu wyjściowego z naniesionymi liniami oddzielającymi poszczególne wiersze tekstu. Architektura opiera się na dwóch rywalizujących ze sobą sieciach

3. **Zbiory danych** 
	Model przetestowano na dwóch standardowych zestawach danych:

	- **HIT-MW:** Baza odręcznego tekstu w języku chińskim.
	- **ICDAR 2013:** Zbiór konkursowy (Handwritten Segmentation Contest) zawierający rękopisy w języku angielskim, greckim i bengalskim.
    
4. **Wyniki i wnioski**
- **Skuteczność:** Model z generatorem U-Net i połączeniem trzech funkcji straty (cGAN + L1 + L2) osiągnął najwyższą skuteczność, przewyższając konkurencyjne metody (tzw. _state-of-the-art_).
    
- **Precyzja:** Uzyskano wskaźnik dokładności (F-measure) na poziomie 99,63% dla zbioru chińskiego (HIT-MW) oraz 98,67% dla wielojęzycznego zbioru ICDAR 2013.
    
- **Znaczenie:** Metoda świetnie radzi sobie z nienormatywnym pismem, nakładającymi się znakami i nieregularnymi odstępami, tworząc solidną podstawę dla wielojęzycznych systemów rozpoznawania tekstu (OCR).