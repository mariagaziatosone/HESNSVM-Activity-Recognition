"""
modules.py - I due modelli usati negli esperimenti: SVM ed ESN.
"""

from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from esn import ESN


def make_svm():
    """Impostazioni del paper. gamma='auto' (= 1/n_feature) va scritto:
    dalla versione 0.22 il default di scikit-learn e' 'scale'."""
    return SVC(kernel="rbf", C=1.0, gamma="auto")


def train_ESN(U, y, Ute, yte, params, seed=0):
    """
    Addestra una ESN sul training (U, y) e la valuta sul test (Ute, yte).
    U, Ute: sequenze (N, passi, canali); params: iperparametri della ESN.
    Restituisce accuracy e F1 pesata, in percentuale.
    """
    esn = ESN(n_channels=U.shape[2], n_classes=10, seed=seed, **params)
    pred = esn.fit(U, y).predict(Ute)
    return (100 * accuracy_score(yte, pred),
            100 * f1_score(yte, pred, average="weighted"))