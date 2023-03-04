# CignoCoords.py
Restituisce le coordinate elevazione e azimuth per la costellazione del cigno. L'azimuth ha già la correzione per l'off-set della parabola.
## CoordsSingle
Se non si passa alcun argomento alla riga di comando `python CignoCoords.py`, viene eseguita la funzione `CoordsSingle`. Essa permette di ottenere le coordinate del cigno `CygCoords` ad un certo giorno e certa ora `time`.
## CoordsBatch
Passando un qualsiasi argomento alla riga di comando `python CignoCoords.py whatever` permette di ottenere la posizione del cigno su più giorni consecutivi alla stessa ora a declinazioni incrementali.

Modificare `coords` per impostare l'intervallo di declinazioni. Modificare `times` per il mese e l'ora di osservazione.

# ElevAzm.py
Restituisce le coordinate elevazione e azimuth per la costellazione del cigno, andromeda e cassiopea: `python ElevAzm.py`. L'azimuth ha già la correzione per l'off-set della parabola.

Modificare le variabili `OggettoCoords` per osservare parti diverse dell'oggetto di interesse ed il rispettivo `time`. Per andromeda non è necessario modificare le coordinate perché la galassia si può osservare tutta dentro il fascio d'antenna.

# Movimenti.py
Passando come argomenti alla linea di comando l'azimuth e l'elevazione `python Movimenti.py azimuth elevazione` permette di sapere se la parabola può puntare in tale direzione.

La funzione `warning_limiti` non funziona ancora come avrei voluto.

## Plot
Se non si passa alcun argomento a linea di comando `python Movimenti.py` si riceve un plot delle posizioni consentite: giallo sì, viola no.

# ReadData.py
Legge i file dal ricevitore digitale della parabola e fa alcuni plot: per ora l'unico importante è quello finale che rappresenta la media temporale in una banda attorno alla frequenza di emissione della linea HI.

Scaricare i file dal Drive `LaboratorioAstro > Ricevitore Digitale > ... > Data 2`. Il fuso orario del ricevitore digitale è quello di Milano, non serve correggere di due ore. Modificare la variabile `data_path` inserendo il percorso dove si trovano i file. La mia struttura delle cartelle è così:
```
.
├── Data
│   ├── Andromeda
│   │   ├── yymmdd_hhmmss_USRP.txt
│   │   └── ...
│   ├── Cassiopea
│   │   ├── yymmdd_hhmmss_USRP.txt
│   │   └── ...
│   └── Cigno
│       ├── yymmdd_hhmmss_USRP.txt
│       └── ...
└── Lab-Astro
    ├── CignoCoords.py
    ├── ElevAzm.py
    ├── Movimenti.py
    ├── ReadData.py
    ├── README.md
    └── test.py
```

Passare a riga di comando il numero ordinale del file partendo da zero (come se fosse un array): `python ReadData.py x`. Chiamare il programma senza numero mostra la lista dei file presenti in ordine cronologico.

# test.py
File per test generici di funzioni, idee e altro.

# Packages richiesti
Questi programmi fanno uso intensivo delle librerie seguenti:
* Astropy
* Matplotlib
* Numpy
* Pandas
* Scipy

Astropy scarica dei file temporanei che utilizza per la conversione da RaDec ad AlzAt.