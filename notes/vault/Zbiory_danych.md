# Najbardziej adekwatne

### Polish Handwritten Characters Database (PHCD)
https://www.kaggle.com/datasets/westedcrean/phcd-polish-handwritten-characters-database
https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-b97bdeca-28ef-4915-a601-b36a819d9ab3 (to chyba do tego)

Baza PHCD, opracowana przez naukowców z Politechniki Lubelskiej, stanowi obecnie najbardziej obszerny fundament dla systemów rozpoznawania polskiej grafii odręcznej. Zbiór ten zawiera ponad 530 000 obrazów znaków, co pozwala na trenowanie głębokich sieci konwolucyjnych (CNN) z wysoką odpornością na różnorodność stylów pisma. Dane te zostały pozyskane od zróżnicowanej demograficznie grupy ponad 2000 uczestników, w tym studentów i pracowników uczelni, co gwarantuje szerokie spektrum wariancji międzyosobniczej.
Struktura techniczna PHCD obejmuje 89 klas decyzyjnych, co pozwala na pełne pokrycie polskiego alfabetu wraz z cyframi i znakami specjalnymi. Każda próbka jest znormalizowanym obrazem w skali szarości o rozdzielczości $32\times32$ piksele.

### PG-Handwritten Letters Dataset (PG-HWLD)
https://mostwiedzy.pl/en/open-research-data/pg-handwritten-letters-dataset-extension-of-emnist-evaluation,930025240940752-0?_share=d1e50a88c594cfb0

Zbiór PG-HWLD, udostępniony przez Politechnikę Gdańską, pełni w inżynierii HTR rolę krytycznego zestawu walidacyjnego, służącego do badania tzw. luki generalizacyjnej. Zawiera on 17 160 próbek odręcznych od 52 osób. Parametry techniczne tego zbioru zostały celowo dostosowane do standardu EMNIST-Letters (rozdzielczość $28\times28$ pikseli, inwersja barw, rozmycie gaussowskie), co umożliwia bezpośrednie porównanie wydajności modeli trenowanych na danych amerykańskich w zderzeniu z polską specyfiką grafomotoryczną.

Analizy przeprowadzone na PG-HWLD wykazują, że nowoczesne modele takie jak VGG-5 czy TextCaps tracą od 11% do 13% dokładności przy przejściu z baz NIST na PG-HWLD, co podkreśla unikalność polskiej dystrybucji pisma nawet w zakresie liter łacińskich. Zbiór ten jest dostępny w repozytorium Most Wiedzy w formacie skompresowanym (PG-HWLD.zip), zawierającym zarówno obrazy w formacie BMP, jak i gotowe skrypty do ich procesowania.

# Specjalistyczne (język medyczny)
### Rejestr Produktów Leczniczych (RPL)
https://dane.gov.pl/pl/dataset/397,rejestr-produktow-leczniczych
https://dane.gov.pl/en/dataset/27,rejestr-systemow-kodowania/resource/10566/table

Zasób administrowany przez Centrum e-Zdrowia (CeZ) jest najważniejszym źródłem danych słownikowych dla projektów OCR/HTR recept. Zawiera on kompletny wykaz leków dopuszczonych do obrotu na terytorium RP. Dane są aktualizowane codziennie i udostępniane publicznie na portalu dane.gov.pl.

### ADMEDVOICE i Korpusy Tekstowe Gdańskiej Nauki
https://www.kaggle.com/datasets/karmarci/polish-medical-multi-source-corpus
https://www.researchgate.net/publication/394518619_A_Comprehensive_Polish_Medical_Speech_Dataset_for_Enhancing_Automatic_Medical_Dictation

Projekt ADMEDVOICE, realizowany m.in. na Politechnice Gdańskiej, dostarcza unikalnych danych z obszaru polskiej terminologii klinicznej. Choć pierwotnie przeznaczony dla systemów rozpoznawania mowy (ASR), zawiera on precyzyjny korpus tekstowy odzwierciedlający żargon używany w gabinetach lekarskich, na oddziałach onkologicznych i w radiologii.   

Zbiór ten obejmuje:
- Ponad 15 godzin autentycznych zapisów tekstowych i audio od 28 lekarzy.   
- Ponad 83 godziny danych syntetycznych (text-to-speech) opartych na realnych dokumentach medycznych.   
- 12.3% wyrazów w korpusie to rygorystyczne terminy medyczne i nazwy leków.   

Wykorzystanie ADMEDVOICE pozwala na trenowanie modeli językowych (np. n-gramów lub mniejszych modeli BERT) w zakresie prawdopodobieństwa współwystępowania słów w kontekście klinicznym.

### Most Wiedzy: Baza Sytuacji Klinicznych
https://mostwiedzy.pl/en/open-research-data/clinical-situations-text-database-for-polish-language,11080602391139881-0

Repozytorium Most Wiedzy udostępnia zbiór zatytułowany „Clinical situations text database for Polish language” (2024). Jest to zestaw ustrukturyzowanych tekstów medycznych podzielonych na 10 kategorii, takich jak wywiad medyczny, badanie onkologiczne, kardiologiczne czy recepty z nazwami farmaceutycznymi. Dane te, zapisane w formacie CSV, stanowią idealny materiał do fine-tuningu modeli HTR w zakresie całych linii tekstu, symulując typowy układ i składnię polskich dokumentów medycznych.

#### Hugging Face: amu-cai/medical-exams-PES-PL-2007-2024
https://huggingface.co/datasets/amu-cai/medical-exams-PES-PL-2007-2024
„Polish Medical Exams”, zawierających ponad 24 000 pytań z egzaminów LEK, LDEK i PES. Choć są to teksty drukowane, ich wartość polega na dostarczeniu bogatego słownictwa z zakresu polskiej medycyny konserwatywnej, stomatologii, chirurgii i prawa medycznego. 


#### Ręcznie labelowane wyrazy z polskich zapisów historycznych (20k)
https://github.com/perechen/htr_lexicography

# Angielskie

### Handwritten Medical Prescriptions Collection (129 recept)
https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset

This dataset comprises a diverse range of medical prescription images obtained from various sources. The prescriptions featured in the dataset exhibit illegible handwriting, commonly encountered in medical practices. These images serve as invaluable resources for developing and evaluating algorithms aimed at enhancing handwriting recognition technologies within the medical domain. Researchers, data scientists, and machine learning enthusiasts can utilize this dataset to train and test models for accurately deciphering illegible medical handwriting, thereby improving patient safety and healthcare efficiency

### Doctor Handwriting Recognition Dataset (90 fragmentów recept + labele)
https://www.kaggle.com/datasets/mrdude20/doctor-handwriting-recognition-dataset

This Doctor Handwriting Dataset contains 90 high-quality handwritten medical prescription samples collected manually from 30 different doctors in Nawabshah, Pakistan. Each medicine name is written by 3 doctors, offering diverse handwriting styles and variations. This dataset is ideal for researchers and developers working on handwriting recognition, optical character recognition (OCR) for medical prescriptions, and AI models focused on medical handwriting analysis. 