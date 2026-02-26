
# Modele OCR

- testować różne gotowe modele 
- podzielić ze względu na typ danych, typ działania, dane treningowe
- wybrać lepsze, które posłużą za benchmarki
- własna architektura dojdzie potem
- zapisywać wyniki w celach porównawczych
- prowadzenie historii i proces selekcji
- początkowo najlepiej na liniach tekstu


# Segmentacja dokumentów

Przede wszystkim:
- ekstrakcja linii tekstu
- wyodrębnianie
Potem:
- transformacje przestrzenne
- preprocessing


# Generowanie danych

Najpierw:
- generowanie (tekstowe) przy pomocy różnych fontów imitujących pismo ręczne
- do tego wszelkie transformacje/deformacje na poziomie zniekształcania znaków, wyrazów, linii
- urealnienie przez szum, odbarwienie, umiejscowienie itp
- można użyć LLM / n-gram cokolwiek będzie mieć sens pod kątem specjalistycznego słownictwa
- uwzględnić zbiory danych (notatka Zbiory_danych) pod kątem języka
- uporządkować ze względu na wszystko. tzn Font, zastosowane przekształcenia, itp

Później:
- rozważyć GAN / stable diffusion pod kątem podkręcenia wizualnego
- Jeśli GAN to ze szczególną uwagą na niepożądany mode collapse do jednego stylu, lub ładnych geometrii pozbawionych treści
- analiza powstałych danych syntetycznych
- porównania wpływu rodzaju danych na skuteczność działania


# Dataset

- przegląd co z dostępnych zbiorów się nadaje
- pozyskanie TEST setu - olabelowane, rzeczywiste zapisy
- przegląd literatury medycznej i potencjalne skanowanie + przepisanie
- zamysł docelowego wyglądu naszego zbioru