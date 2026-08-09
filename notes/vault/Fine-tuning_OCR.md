## Podział Modeli wg Priorytetów Treningowych

### GRUPA 1: Pierwsza Fala (Najwyższy Potencjał Quality/Speed na wycinkach)

#### 1. Surya OCR

- **Typ:** Vision Transformer
    
- **Dlaczego warto:** Zbudowany od podstaw z myślą o wielojęzycznym OCR. Posiada wbudowane wsparcie dla polskich glifów.
    
- **Rola:** Główny koń roboczy. Najniższy próg wejścia do uzyskania wysokiego accuracy na odręcznych polskich wycinkach.
    
- **Strategia FT:** Fine-tuning samego modułu OCR (bez włączonej detekcji).
    

#### 2. PARSeq (`baudm/parseq`)

- **Typ:** Lightweight ViT / Permutation Language Model
    
- **Dlaczego warto:** Jeden z najnowocześniejszych modeli do OCR słów/wycinków. Błyskawiczny i lekki.
    
- **Uwaga:** Wersja bazowa nie zna polskich znaków.
    
- **Strategia FT:** Wymagana zmiana parametru `charset` w konfiguracji (dodanie `ąćęłńóśźż`) i trening na własnych syntetykach. Idealny kandydata do produkcji o wysokiej przepustowości.
    

#### 3. TrOCR (`microsoft/trocr-base-handwritten`)

- **Typ:** Vision-Encoder-Decoder (ViT + GPT-2/RoBERTa)
    
- **Dlaczego warto:** Architektura sekwencja-do-sekwencji świetnie radzi sobie ze skrajnie nieczytelną kursywą.
    
- **Uwaga:** Model bazowy "sprawia wrażenie, że nie działa na PL", ponieważ jego dekoder ma zafiksowane angielskie słownictwo i zamienia polskie wyrazów na angielskie odpowiedniki.
    
- **Strategia FT:** Trening z odpowiednim `learning_rate` nadpisze wagi dekodera i nauczy model polskich struktur językowych.
    

### GRUPA 2: Baseline i Optymalizacja Wydajnościowa (CPU / Mobile)

#### 4. Tesseract (`tesseract_pol`)

- **Typ:** CRNN / LSTM + CTC
    
- **Zalety:** Znikome wymagania sprzętowe, inferencja na CPU.
    
- **Ograniczenia:** Gorzej radzi sobie z ciągłą kursywą i zmienną grubością kreski z powodu binarizacji i ograniczeń architektonicznych CTC.
    
- **Strategia FT:** Użycie skryptów `tesstrain` do dociągnięcia do polskich danych. Służy jako **production baseline** do mierzenia czy wyższa jakość z Transformerów uzasadnia większe koszty GPU.
    

#### 5. PaddleOCR (`PP-OCRv4_mobile_rec`)

- **Typ:** SVTR / Mobile-CRNN
    
- **Zalety:** Lekki, szybki, nowocześniejszy od EasyOCR i Tesseracta.
    
- **Strategia FT:** Dobry kandydat na szybki backend produkcyjny, jeśli Tesseract okaże się niewystarczająco dokładny.
    

### GRUPA 3: "Górna Półka" Jakościowa (Gdy priorytetem jest jakość, a nie koszt VRAM)

#### 6. Qwen2.5-VL-3B-Instruct

- **Typ:** Large Vision-Language Model (VLM)
    
- **Zalety:** Wybitna zdolność "odgadywania" trudnego tekstu z kontekstu semantycznego słowa/zdania.
    
- **Ograniczenia:** Znacznie większe zapotrzebowanie na VRAM i wolniejsze wnioskowanie.
    
- **Strategia FT:** Fine-tuning przez QLoRA / Unsloth z naciskiem na krótki prompt (np. _"Rozpoznaj tekst z wycinka:"_).
    

### GRUPA 4: Niezalecane / Niski Priorytet dla Wycinków Tekstu

- **GOT-OCR 2.0 & GLM-4V-9B:** Zaprojektowane do architektury _Document-level_ (całe arkusze, schematy, tabele). Ich użycie do małych cropów linii/słów to nieefektywne gospodarowanie zasobami GPU.
    
- **EasyOCR:** Starsza architektura ResNet+BiLSTM+CTC. W testach ustępuje PARSeq oraz PaddleOCR.
    
- **RysOCR:** Warto sprawdzić jedynie wtedy, gdy adapter na PaddleOCR-VL wniesie istotną wartość ponad czysty PaddleOCR, choć przy samych cropach zysk może być znikomy.


|**Model**|**Architektura**|**Zapotrzebowanie VRAM**|**Szybkość Inferencji**|**Potencjał HTR (PL)**|**Priorytet Treningowy**|
|---|---|---|---|---|---|
|**Surya OCR**|ViT|Średnie|Szybka|**Bardzo wysoki**|**1 (Top)**|
|**PARSeq**|ViT + Autoregressor|Niskie|Bardzo szybka|**Wysoki** (po zmianie charsetu)|**1**|
|**TrOCR**|ViT + Decoder|Średnie|Średnia|**Wysoki** (po fine-tuning)|**1**|
|**Tesseract_pol**|LSTM + CTC|Znikome (CPU)|Błyskawiczna|**Średni** (Słabszy na kursywie)|**2 (Baseline)**|
|**PaddleOCR**|SVTR / CRNN|Niskie|Bardzo szybka|**Średni / Wysoki**|**2**|
|**Qwen2.5-VL-3B**|VLM|Wysokie|Wolna|**Ekstremalnie wysoki**|**3 (Górna półka)**|
|**RysOCR**|Paddle-VL Adapter|Średnie|Średnia|**Do weryfikacji**|**4**|
|**EasyOCR**|ResNet + CRNN|Niskie|Średnia|**Niski** (Koncepcja starzejąca się)|**5**|
|**GOT-OCR 2.0**|VLM / ViT|Wysokie|Wolna|**Niski** (Overkill na cropy)|**5**|
|**GLM-4V-9B**|Large VLM|Bardzo wysokie|Bardzo wolna|**Niski** (Za duży na cropy)|**5**|