"""
main.py - Esperimenti della consegna (punti 2, 3 e 4).

Esecuzione (dalla cartella code/):   python main.py
Il grafico della Fig. 5 viene salvato in ../imgs/esn_fig5.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # salva i grafici su file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_dataset, make_preprocessor, preprocess_all
from esn import encode_input
from modules import make_svm, train_ESN

SEEDS = [0, 1, 2]                                          # tre seed diversi
MODES = ["single_step", "feature_as_sequence", "repeat"]   # codifiche dello scheletro

# Parametri della sez. 5.2 del paper (nomi dello scheletro, vedi esn.py)
ESN_SEC52 = dict(n_reservoir=700, sparsity=0.33, spectral_radius=1.3,
                 input_scaling=1.7, noise=1e-5, ridge=15.12)
# Parametri della Fig. 5 del paper (si varia solo spectral_radius)
ESN_FIG5 = dict(n_reservoir=700, sparsity=0.33, input_scaling=0.47,
                noise=0.7, ridge=2.12)

# ------ Dati ------
X, y, groups, class_names = load_dataset()
X18 = preprocess_all(X, y)          # 18 feature, preprocessing globale (come il paper)


def esn_70_30(U, params):
    """ESN con split stratificato 70/30, ripetuta con i tre seed.
    Restituisce la media di (accuracy, F1)."""
    results = []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                                  stratify=y, random_state=seed)
        results.append(train_ESN(U[tr], y[tr], U[te], y[te], params, seed))
    return np.mean(results, axis=0)


def dataset_check():
    """Punto 2: il dataset corrisponde al paper?"""
    print("\n===== Dataset =====")
    print(f"campioni {X.shape[0]} (paper 1150), feature {X.shape[1]} (paper 105), "
          f"classi {len(class_names)} (paper 10)")
    print("valori mancanti:", X.isna().sum().sum())
    print("campioni per classe:", dict(zip(class_names, np.bincount(y).tolist())))
    print("partecipanti:", pd.Series(groups).value_counts().sort_index().to_dict())
    pre = make_preprocessor().fit(X, y)
    print(f"dopo le colonne duplicate: {pre[0].keep_.sum()} feature (paper 35), "
          f"dopo SelectKBest: {pre[1].get_support().sum()} (paper 18)")


def svm_iris():
    """Punto 3: SVC su iris al variare di C e gamma, con e senza scaler."""
    print("\n===== SVC su iris (5-fold stratificata) =====")
    Xi, yi = load_iris(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    gammas = [0.01, 0.1, 1, 10]
    for use_scaler in [False, True]:
        print("con StandardScaler" if use_scaler else "senza scaler")
        print("C \\ gamma " + "".join(f"{g:>8}" for g in gammas))
        for C in [0.1, 1, 10, 100]:
            row = f"{C:<10}"
            for g in gammas:
                model = SVC(kernel="rbf", C=C, gamma=g)
                if use_scaler:
                    model = make_pipeline(StandardScaler(), model)
                row += f"{cross_val_score(model, Xi, yi, cv=cv).mean():8.3f}"
            print(row)

    # Multiclasse: il paper parla di one-vs-rest. Con 'ovo' SVC mostra una
    # colonna per ogni classificatore binario che addestra: 10*9/2 = 45
    # coppie di classi, cioe' one-vs-one. 'ovr' riaggrega solo i voti.
    m = SVC(kernel="rbf", C=1, gamma="auto", decision_function_shape="ovo").fit(X18, y)
    print(f"classificatori binari addestrati da SVC: {m.decision_function(X18[:1]).shape[1]}"
          " -> one-vs-one, non one-vs-rest")


def svm_paper():
    """Punti 2-3: SVM del paper, 2 validazioni x 2 varianti di preprocessing."""
    print("\n===== SVM del paper (atteso: ~91% accuracy, ~90% F1) =====")
    schemes = {
        "10-fold stratificata, seed 0": (StratifiedKFold(10, shuffle=True, random_state=0), None),
        "partecipanti separati":        (GroupKFold(10), groups),
    }
    for name, (cv, g) in schemes.items():
        for variant in ["globale", "nel fold"]:
            if variant == "globale":       # preprocessing gia' fatto su tutto il dataset
                data, model = X18, make_svm()
            else:                          # preprocessing rifatto dentro ogni fold
                data, model = X, make_pipeline(make_preprocessor(), make_svm())
            r = cross_validate(model, data, y, cv=cv, groups=g,
                               scoring=["accuracy", "f1_weighted"])
            print(f"{name:30s} {variant:9s} accuracy {100 * r['test_accuracy'].mean():6.2f}"
                  f"   F1 {100 * r['test_f1_weighted'].mean():6.2f}")


def esn_sec52():
    """Punto 4: parametri della sez. 5.2, split 70/30, tre seed."""
    print("\n===== ESN, sez. 5.2, split 70/30, 3 seed (paper: 92.08 / 91.90) =====")
    for mode in MODES:
        acc, f1 = esn_70_30(encode_input(X18, mode), ESN_SEC52)
        print(f"{mode:20s} accuracy {acc:6.2f}   F1 {f1:6.2f}")


def esn_fig5():
    """Punto 4 / scheletro: effetto di rho per ogni codifica (Fig. 5 del paper)."""
    print("\n===== Fig. 5: accuracy al variare del raggio spettrale =====")
    rhos = [0.1, 0.5, 1.0, 1.3, 1.7, 2.0, 3.0]
    paper = [24.5, 42.5, 72.5, 79.5, 75, 61, 31]           # letti dal grafico del paper
    table = pd.DataFrame({"rho": rhos, "paper": paper})
    for mode in MODES:
        U = encode_input(X18, mode)
        accs = []
        for rho in rhos:
            acc, _ = esn_70_30(U, {**ESN_FIG5, "spectral_radius": rho})
            accs.append(round(acc, 1))
        table[mode] = accs
    print(table.to_string(index=False))

    plt.figure(figsize=(6, 3.5))
    for mode in MODES:
        plt.plot(rhos, table[mode], "o-", label=mode)
    plt.plot(rhos, paper, "k--", label="paper (Fig. 5)")
    plt.xlabel("raggio spettrale"); plt.ylabel("accuracy (%)")
    plt.legend(); plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "imgs"
    out.mkdir(exist_ok=True)
    plt.savefig(out / "esn_fig5.png", dpi=120)


if __name__ == "__main__":
    dataset_check()
    svm_iris()
    svm_paper()
    esn_sec52()
    esn_fig5()