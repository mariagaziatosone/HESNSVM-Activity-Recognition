"""
modules.py - Funzioni di addestramento dei modelli.

Ogni train_* riceve training e test, addestra e restituisce
(accuracy, f1) in percentuale, come train_ESN del codice BDESN.
I parametri della ESN usano i nomi dello scheletro (esn.py):
    n_reservoir, sparsity, spectral_radius, input_scaling, noise, ridge
"""

import warnings

from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from esn import ESN

# scikit-learn >= 1.9 avvisa che SVC(probability=True) sara' rimosso in 1.11:
# per ora funziona, nascondiamo solo l'avviso.
warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)


def scores(y_true, y_pred):
    """Accuracy e F1 pesata (come nel paper), in percentuale."""
    return (100 * accuracy_score(y_true, y_pred),
            100 * f1_score(y_true, y_pred, average="weighted"))


def make_svm(probability=False, seed=0):
    """Impostazioni del paper. gamma='auto' (= 1/n_feature) va scritto:
    dalla versione 0.22 il default di scikit-learn e' 'scale'."""
    return SVC(kernel="rbf", C=1.0, gamma="auto",
               probability=probability, random_state=seed)


def train_SVM(X, y, Xte, yte):
    model = make_svm().fit(X, y)
    return scores(yte, model.predict(Xte))


def train_ESN(U, y, Ute, yte, params, seed=0):
    """U, Ute: sequenze (N, passi, canali); params: iperparametri della ESN."""
    esn = ESN(n_channels=U.shape[2], n_classes=10, seed=seed, **params)
    esn.fit(U, y)
    return scores(yte, esn.predict(Ute))


def train_HESNSVM(U, y, Ute, yte, params, seed=0):
    """
    HESNSVM (sez. 3 del paper):
      1. stati del reservoir S (gli stessi per i due classificatori)
      2. readout ridge su S -> L (logit)      SVC su S -> P (probabilita')
      3. O = P * L elemento per elemento, classe = argmax(O)
    Restituisce i punteggi di ESN, SVM sugli stati e HESNSVM.
    """
    esn = ESN(n_channels=U.shape[2], n_classes=10, seed=seed, **params)
    S, Ste = esn.run_reservoir(U), esn.run_reservoir(Ute)

    L = esn.fit_states(S, y).decision_from_states(Ste)   # (N_test, 10)
    svm = make_svm(probability=True, seed=seed).fit(S, y)
    P = svm.predict_proba(Ste)                           # (N_test, 10)

    return {"ESN": scores(yte, L.argmax(1)),
            "SVM (stati)": scores(yte, P.argmax(1)),
            "HESNSVM": scores(yte, (P * L).argmax(1))}
