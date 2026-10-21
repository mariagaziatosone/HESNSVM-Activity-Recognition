"""
data_utils.py - Caricamento del dataset e preprocessing.

Pipeline del paper (sez. 4.3-4.4):
    105 feature -> togli le colonne duplicate -> 35
                -> SelectKBest(k=18)          -> 18
                -> StandardScaler             -> 18 standardizzate
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

URL = ("https://raw.githubusercontent.com/LIARALab/Datasets/master/"
       "ActivitiesExercisesRecognition/features/9_axis_imu_lsm9ds1.csv")
CSV = Path(__file__).resolve().parent.parent / "dataset" / "9_axis_imu_lsm9ds1.csv"


def load_dataset():
    """
    Scarica il CSV la prima volta, poi lo legge.
    Restituisce:
        X       DataFrame (1150, 105) delle feature
        y       classi come interi 0..9
        groups  partecipante di ogni riga (lettere A..J)
        names   nome di ogni classe (names[0] e' la classe 0, ecc.)
    """
    if not CSV.exists():
        CSV.parent.mkdir(exist_ok=True)
        print("Scarico il dataset da", URL)
        urllib.request.urlretrieve(URL, CSV)

    df = pd.read_csv(CSV)                       # il file ha l'intestazione
    X = df.drop(columns=["action", "participant"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["action"])
    return X, y, df["participant"].to_numpy(), list(encoder.classes_)


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    """
    Passo 105 -> 35. Il paper non dice la regola: noi togliamo ogni colonna
    con valori IDENTICI a una colonna precedente. Nel file i blocchi
    Giroscopio e Magnetometro sono copie dell'Accelerometro, quindi
    restano esattamente 35 colonne.
    E' un trasformatore di scikit-learn, cosi' entra in una Pipeline.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(np.asarray(X))
        self.keep_ = ~X.T.duplicated().to_numpy()    # True = colonna da tenere
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def make_preprocessor(k=18):
    """Duplicate -> SelectKBest (punteggio f_classif, il default) -> scaler."""
    return make_pipeline(RemoveDuplicates(),
                         SelectKBest(f_classif, k=k),
                         StandardScaler())


def preprocess_all(X, y, k=18):
    """
    Variante "globale" (come sembra fare il paper): il preprocessing vede
    TUTTO il dataset, anche le righe che poi finiranno nel test.
    La variante "nel fold" invece mette make_preprocessor() in una Pipeline.
    """
    return make_preprocessor(k).fit_transform(X, y)