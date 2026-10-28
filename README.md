# Riproduzione di HESNSVM per il riconoscimento delle attività umane

Riproduzione degli esperimenti del paper:

> S. Abudalfa, K. Bouchard, *Hybrid Deep-Readout Echo State Network and Support Vector Machine with Feature Selection for Human Activity Recognition*, BDTA/WiCON 2020, LNICST 371, pp. 150–167, Springer, 2021. https://doi.org/10.1007/978-3-030-72802-1_11

---

## 1. Obiettivi del progetto

### Il problema
Una persona indossa al polso un sensore inerziale a 9 assi (accelerometro, giroscopio, magnetometro) e svolge diverse attività: cammina, corre, si siede, fa uno squat. L'obiettivo è **riconoscere automaticamente l'attività** a partire dai dati del sensore.

### Il paper
Abudalfa e Bouchard propongono **HESNSVM**, un modello che combina:
- una **Echo State Network (ESN)**: una rete neurale ricorrente in cui solo lo strato d'uscita viene addestrato (regressione ridge), mentre il "reservoir" interno resta casuale e fisso;
- una **Support Vector Machine (SVM)**.

Il funzionamento (sez. 3 del paper):
1. **in ingresso** arrivano le 18 feature standardizzate di ogni campione;
2. il reservoir trasforma ogni campione in uno stato ad alta dimensione;
3. sugli stati si addestrano il readout ridge, che produce i *logit* **L**, e una SVM, che produce le probabilità **P**;
4. **decisione finale**: O = P × L, e la classe predetta è `argmax(O)`;
5. **in uscita** si ottiene una delle 10 attività.

### Il dataset
[LIARA ActivitiesExercisesRecognition](https://github.com/LIARALab/Datasets/tree/master/ActivitiesExercisesRecognition), file `features/9_axis_imu_lsm9ds1.csv` (Chapron et al., *Sensors* 18, 268, 2018):
- **1150 campioni**, **10 attività**, **10 partecipanti** (A–J, 115 campioni ciascuno);
- **105 feature** già estratte dai segnali (statistiche nel tempo e in frequenza).

### Obiettivi
1. **Preparazione**: ambiente Python isolato e repository Git, con le versioni delle librerie salvate.
2. **Dataset e preprocessing**: verificare il dataset e riprodurre la pipeline del paper (105 → 35 → 18 feature → StandardScaler), in due varianti: preprocessing su tutto il dataset e preprocessing dentro ogni fold.
3. **SVM**: provare SVC su un dataset giocattolo (iris), poi valutare la SVM del paper (rbf, C = 1, `gamma='auto'`) con una 10-fold stratificata e con i partecipanti separati tra i fold.
4. **ESN da zero in NumPy**, partendo dallo scheletro fornito: scegliere come le 18 feature entrano nel reservoir e verificare i risultati della sez. 5.2 (circa 92%).
5. **BDESN**: far girare il codice di riferimento (Python 3.7, TensorFlow 1.14) in un ambiente separato.

---

## 2. Spiegazione dei file del codice

### Struttura
```
hesnsvm/
├── code/
│   ├── data_utils.py   caricamento del dataset e preprocessing
│   ├── esn.py          Echo State Network in NumPy (scheletro completato) + 5 test
│   ├── modules.py      creazione della SVM e addestramento della ESN
│   └── main.py         esegue tutti gli esperimenti
├── dataset/            il CSV, scaricato automaticamente al primo avvio
├── imgs/               grafici prodotti
└── requirements.txt    versioni delle librerie (pip freeze)
```
Collegamenti tra i file: `main.py` usa `modules.py`, che usa `esn.py`; i dati arrivano da `data_utils.py`.

### `data_utils.py` — dati e preprocessing
| Elemento | Cosa fa |
|---|---|
| `load_dataset()` | al primo avvio scarica il CSV da GitHub in `dataset/`, poi lo legge. Restituisce le 105 feature, la classe (0–9), il partecipante (A–J) e i nomi delle classi |
| `RemoveDuplicates` | **105 → 35**: elimina le colonne identiche a una colonna precedente. Nel file pubblicato i blocchi giroscopio e magnetometro sono copie esatte dell'accelerometro, quindi restano esattamente 35 colonne |
| `make_preprocessor()` | la pipeline del paper: `RemoveDuplicates` → `SelectKBest(f_classif, k=18)` → `StandardScaler` |
| `preprocess_all()` | applica la pipeline a tutto il dataset prima della validazione (variante "globale", come nel paper) |

### `esn.py` — la Echo State Network (scheletro fornito, completato)
| Elemento | Cosa fa |
|---|---|
| `encode_input(X, mode)` | trasforma le 18 feature nella sequenza da dare al reservoir: `single_step` (un passo con 18 canali), `feature_as_sequence` (18 passi con un canale, **scelta**) oppure `repeat` (lo stesso vettore ripetuto) |
| `make_reservoir_matrix()` | crea la matrice ricorrente W: pesi uniformi in [-0,5, 0,5], poi la maschera di sparsità β, poi il riscalamento al raggio spettrale ρ |
| `ESN.__init__` | pesi di input W_in uniformi in [-1, 1] × ω, e la matrice W |
| `ESN.run_reservoir(U)` | aggiorna lo stato a ogni passo con `x(t) = tanh(W_in·u(t) + W·x(t−1) + rumore)` e restituisce l'ultimo stato |
| `ESN.fit(U, y)` | readout con regressione ridge su target one-hot, risolta con `np.linalg.solve` in forma primale o duale |
| `ESN.decision_function` / `predict` | i 10 punteggi del readout e la classe con il punteggio più alto |
| `test_*` (5 test) | raggio spettrale corretto; stati limitati (con grafico); echo state property; readout identico a `sklearn.linear_model.Ridge`; la ricorrenza conta (con W = 0 l'accuratezza crolla) |
| blocco `__main__` | esempio minimo: parametri della sez. 5.2, split 70/30, 3 seed |

### `modules.py` — i modelli
| Elemento | Cosa fa |
|---|---|
| `make_svm()` | `SVC(kernel="rbf", C=1.0, gamma="auto")`. `gamma='auto'` è scritto esplicitamente perché dalla versione 0.22 il default di scikit-learn è `'scale'` |
| `train_ESN(...)` | addestra una ESN sul training e restituisce accuratezza e F1 pesata sul test |

### `main.py` — gli esperimenti
| Funzione | Cosa fa |
|---|---|
| `dataset_check()` | campioni, feature, classi, valori mancanti, campioni per classe e per partecipante, numero di feature dopo ogni passo |
| `svm_iris()` | SVC su iris al variare di C e gamma, con e senza scaler; verifica della strategia multiclasse |
| `svm_paper()` | SVM del paper con la 10-fold stratificata e con i partecipanti separati, ciascuna con preprocessing globale e dentro il fold |
| `esn_sec52()` | ESN con i parametri della sez. 5.2, split 70/30, 3 seed, per ogni codifica dell'input |
| `esn_fig5()` | accuratezza al variare di ρ per ogni codifica, confrontata con la Fig. 5 del paper; salva `imgs/esn_fig5.png` |

### Esecuzione
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn matplotlib pytest
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
cd code
..\.venv\Scripts\python.exe -m pytest -v esn.py   # 5 test
..\.venv\Scripts\python.exe esn.py               # esempio minimo, sez. 5.2
..\.venv\Scripts\python.exe main.py              # tutti gli esperimenti (~1 minuto)
```

BDESN è eseguito a parte, in un ambiente conda con Python 3.7 e TensorFlow 1.14, usando il [codice di riferimento di Bianchi et al.](https://github.com/FilippoMB/Bidirectional-Deep-readout-Echo-State-Network) clonato in una cartella separata.

---

## 3. Conclusioni e risultati ottenuti

### Dataset
Il dataset corrisponde al paper: **1150 campioni, 10 classi, 105 feature, nessun valore mancante**. Le classi hanno 100 campioni ciascuna, tranne corsa, seduta e camminata che ne hanno 150. L'identificativo del partecipante è presente (10 persone, 115 campioni ciascuna), e il file è ordinato per partecipante.

### SVM (rbf, C = 1, `gamma='auto'`)
| Validazione | Preprocessing | Accuratezza | F1 |
|---|---|---|---|
| 10-fold stratificata, seed 0 | globale | 92,96 | 92,82 |
| 10-fold stratificata, seed 0 | nel fold | 92,96 | 92,82 |
| Partecipanti separati (GroupKFold) | globale | **90,87** | **89,58** |
| Partecipanti separati (GroupKFold) | nel fold | 90,70 | 89,44 |
| *Paper (Tab. 7)* | | *90,9* | *89,6* |

- **Preprocessing globale o dentro il fold**: la differenza è al massimo 0,2 punti, quindi su questo dataset il data leakage del preprocessing globale è trascurabile.
- **Partecipanti separati**: si ottiene esattamente il risultato del paper, e i valori dei singoli fold coincidono con la Tab. 7. Il paper ha quindi valutato con un partecipante per fold, perché il file è ordinato per partecipante e non è stato mescolato.
- **Stratificata mescolata**: il risultato sale di circa 2 punti, perché campioni della stessa persona finiscono sia nel training sia nel test.

### ESN (sez. 5.2: Nr = 700, β = 0,33, ρ = 1,3, ω = 1,7, £ = 0,00001, λ = 15,12; split 70/30, 3 seed)
| Codifica dell'input | Accuratezza | F1 |
|---|---|---|
| `single_step` | 94,78 | 94,71 |
| **`feature_as_sequence`** | **91,69** | **91,53** |
| `repeat` | 94,40 | 94,30 |
| *Paper* | *92,08* | *91,90* |

**Scelta: `feature_as_sequence`** (una feature per passo). È la più vicina al risultato del paper, ed è l'unica che riproduce la Fig. 5. Con `single_step` lo stato iniziale è nullo e c'è un solo passo, quindi la matrice ricorrente non interviene mai e ρ non ha effetto: l'accuratezza resta piatta al 92% per ogni ρ. Con `feature_as_sequence` si ottiene invece la stessa curva a campana del paper (28% → 79% → 31%, contro 25% → 80% → 31%).

Le verifiche richieste sono superate: il raggio spettrale misurato è uguale a quello impostato (1,300000), gli stati restano in [-1, 1], e la echo state property vale anche con ρ = 1,3.

### BDESN
Il codice di riferimento gira in un ambiente conda separato (Python 3.7, TensorFlow 1.14.0, numpy 1.16.4, scikit-learn 0.21.3). Per farlo funzionare è stato necessario fissare `gast==0.2.2` e `protobuf==3.20.3`. Prova sul dataset JAP incluso nel repository, con 500 epoche e 2 run:

| ESN | GRU | BDESN |

|---|---|---|
| 75,4% | 84,2% | 90,7% |

### Decisioni prese dove il paper non dà indicazioni
1. **105 → 35**: si eliminano le colonne duplicate.
2. **SelectKBest**: punteggio `f_classif`, quello predefinito.
3. **Codifica dell'input**: una feature per passo.
4. **ESN**, allineata al codice di riferimento usato dagli autori:
   - W_in uniforme in [-1, 1] × ω;
   - W uniforme in [-0,5, 0,5], prima la maschera e poi il riscalamento;
   - rumore uniforme dentro la tanh;
   - nessun bias, stato iniziale nullo, al readout va l'ultimo stato;
   - intercetta del ridge non regolarizzata.

### Punti poco chiari del paper
1. La regola per passare da 105 a 35 feature non è spiegata, e il dataset pubblicato contiene colonne duplicate.
2. La "10-fold cross validation" non è mescolata: con il file ordinato equivale a lasciare fuori un partecipante per fold.
3. Il paper parla di SVM one-versus-rest, ma `SVC` di scikit-learn addestra 45 classificatori **one-vs-one**.
4. Non viene detto come le 18 feature entrano nel reservoir; lo si deduce solo dalla Fig. 5.
5. Il rumore usato per la Fig. 5 è 0,7 nel paper e 0,07 nel codice di riferimento.

### Limiti
- Jupyter non è installabile con Python 3.15: la sua dipendenza `pyyaml` non ha ancora un pacchetto precompilato per questa versione.
- BDESN è stato provato solo sul dataset di esempio del repository, non sul nostro dataset.
