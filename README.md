# Seminarski rad iz TOVI-TFAI

- *11.* poglavlje iz knjige Luger-a ''Machine Learning: Connection List''. 
    - Temelji konekcionističkih mreža
        - Rana istorija
    - Perceptron obuka
        - Algoritam
        - Primer: Korišćenje mreže perceptrona za klasifikaciju
        - Uopšteno delta pravilo
    - Backpropagation obuka
        - Backpropagation primer 1 : NETtalk
        - Backpropagation primer 2: XOR - ekskluzivna disjunkcija
    - Takmičarsko obučavanje
        - ”Pobednik uzima sve” algoritam klasifikacije za obuku
        - Kohonenova mreža za prototipsko obučavanje
        - Outstar mreže i suprotna propagacija
        - Mašine potpornih vektora - SVM (Harrison, Luger ’02.)
    - Hebova obuka slučajnosti
        - Nenadgledano Hebbian obučavanje
        - Nadgledana Hebbian obuka
        - Asocijativna memorija i linearni veznik
    - Atraktivne mreže ili ”Uspomene”
        - BAM - Dvosmerna asocijativna memorija
        - Autoasociativna memorija and Hopfield mreže

### Implementacija : 
- Backpropagation algorithm via MLP over Iris dataset `./src/main.py`
    - Dokumentacija : 39. strana u `seminarski.pdf`
    - Okruženje nad kojim je testiran projekat je `Anaconda3-2023.09-0...` i `Spyder IDE 5.5.0`, `python 3.11`,
    - Ekstenzije korišćene:
        ```
        matplotlib
        numpy
        pandas
        scikit-learn
        ```
    - Ako i dalje ne štima nešto videti/upotrebiti (pri importu kao environment) `conda-reqs.yaml`.
- Dataset : Iris
- Mentor : dr. Tatjana Stojanović

## Extra : Razvoj i oblasti veštačke inteligencije 
- Diskusija na 1. poglavlje knjige Luger-a
    - Uvod
        - Pogled na pojmove inteligencije, znanja, ljudske imitacije
        - Istorija temelja veštačke inteligencije
        - Tjuringov test
        - Biološki i socijalni model inteligencije: Teorije agenata
    - Pregled oblasti primena veštačke inteligencije
        - Igranje igara
        - Automatsko razumevanje i pretraga dokaza teorema
        - Ekspert sistemi
        - Razumevanje i semantika prirodnih jezika
        - Modeliranje ljudskog ophodenja
        - Planiranje i robotika
        - Jezik i sredine namenjene veštačkoj inteligenciji
        - Mašinsko učenje
        - Alternativne reprezentacije: Neuronske mreže i genetski algoritmi
        - Veštačka inteligencija i filozofije
  
- Knjiga : Artificial Intelligence: Structures and Strategies for Complex Problem Solving 6th Edition 