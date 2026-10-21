"""
main.py - Riproduzione di HESNSVM (Abudalfa & Bouchard, 2021).

Esecuzione (dalla cartella code/):   python main.py
Per scegliere cosa eseguire, modifica gli interruttori qui sotto.
Tabelle in ../results/, grafici in ../imgs/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # salva i grafici su file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.datasets import load_iris
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import load_dataset, make_preprocessor, preprocess_all
from modules import make_svm, train_ESN, train_HESNSVM, train_SVM
from esn import ESN, encode_input

# ------ Cosa eseguire ------
RUN_DATASET_CHECK = True    # passo 2: il dataset e' quello del paper?
RUN_SVM_IRIS = True         # passo 3: SVC su un dataset giocattolo
RUN_SVM = True              # passo 3: SVM con le impostazioni del paper
RUN_ESN = True              # passo 4: ESN, parametri della sez. 5.2
RUN_FIG5 = True             # passo 4: effetto del raggio spettrale (Fig. 5)
RUN_HESNSVM = True          # facoltativo: il modello completo
EXPORT_BDESN = True         # passo 5: dati per il codice BDESN

SEEDS = [0, 1, 2]           # un seed per run: split e reservoir
MODES = ["single_step", "feature_as_sequence", "repeat"]   # codifiche dello scheletro

# ------ Iperparametri ------
# Sez. 5.2 del paper (split 70/30)
# (nomi dei parametri dello scheletro: vedi la classe ESN in esn.py)
ESN_SEC52 = dict(n_reservoir=700, sparsity=0.33, spectral_radius=1.3,
                 input_scaling=1.7, noise=1e-5, ridge=15.12)
# Fig. 5 del paper (si varia solo spectral_radius)
ESN_FIG5 = dict(n_reservoir=700, sparsity=0.33, input_scaling=0.47,
                noise=0.7, ridge=2.12)
# Tabella 6 del paper (10-fold)
ESN_TAB6 = dict(n_reservoir=1500, sparsity=0.2758, spectral_radius=1.3130,
                input_scaling=0.4319, noise=0.0217, ridge=8.6215)

ROOT = Path(__file__).resolve().parent.parent
RESULTS, IMGS = ROOT / "results", ROOT / "imgs"
RESULTS.mkdir(exist_ok=True)
IMGS.mkdir(exist_ok=True)

# ------ Dati (usati da tutti gli esperimenti) ------
X, y, groups, class_names = load_dataset()
X18 = preprocess_all(X, y)                  # preprocessing globale, come il paper
U_seq = encode_input(X18, "feature_as_sequence")   # (1150, 18, 1): codifica scelta


def save_versions():
    """Passo 1: salva le versioni di Python e delle librerie usate."""
    import platform
    import sklearn
    import scipy
    import matplotlib
    lines = [f"python {platform.python_version()}"] + [
        f"{m.__name__} {m.__version__}" for m in (np, pd, scipy, sklearn, matplotlib)]
    (RESULTS / "versioni.txt").write_text("\n".join(lines) + "\n")
    print("versioni:", ", ".join(lines))


def split_70_30(seed):
    """Indici di training e test, split stratificato 70/30."""
    return train_test_split(np.arange(len(y)), test_size=0.3,
                            stratify=y, random_state=seed)


def dataset_check():
    print("\n===== Dataset =====")
    print(f"campioni {X.shape[0]} (paper 1150), feature {X.shape[1]} (paper 105), "
          f"classi {len(class_names)} (paper 10), valori mancanti {X.isna().sum().sum()}")
    print("campioni per classe:", dict(zip(class_names, np.bincount(y).tolist())))
    print("campioni per partecipante:", pd.Series(groups).value_counts().sort_index().to_dict())
    ordered = all(len(set(groups[i * 115:(i + 1) * 115])) == 1 for i in range(10))
    print("righe ordinate per partecipante (blocchi da 115):", ordered)
    acc, gyro, mag = X.iloc[:, :35].values, X.iloc[:, 35:70].values, X.iloc[:, 70:].values
    print("Giroscopio == Accelerometro:", np.array_equal(acc, gyro),
          "| Magnetometro == Accelerometro:", np.array_equal(acc, mag))
    pre = make_preprocessor().fit(X, y)
    n35 = pre[0].keep_.sum()
    chosen = X.columns[pre[0].keep_][pre[1].get_support()]
    print(f"dopo le duplicate: {n35} feature; dopo SelectKBest: {len(chosen)}")
    print("feature scelte:", list(chosen))


def svm_iris():
    print("\n===== SVC su iris (5-fold stratificata) =====")
    Xi, yi = load_iris(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    gammas = [0.01, 0.1, 1, 10, "auto", "scale"]
    for use_scaler in [False, True]:
        print("con StandardScaler" if use_scaler else "senza scaler")
        print("C \\ gamma " + "".join(f"{str(g):>8}" for g in gammas))
        for C in [0.1, 1, 10, 100]:
            row = f"{C:<10}"
            for g in gammas:
                model = SVC(kernel="rbf", C=C, gamma=g)
                if use_scaler:
                    model = make_pipeline(StandardScaler(), model)
                row += f"{cross_val_score(model, Xi, yi, cv=cv).mean():8.3f}"
            print(row)
    # Multiclasse: il paper parla di one-vs-rest. Verifica sui NOSTRI dati
    # (10 classi): con 'ovo' escono 10*9/2 = 45 colonne, una per coppia di
    # classi = i 45 classificatori binari one-vs-one che libsvm addestra
    # sempre. 'ovr' (il default) riaggrega soltanto i loro voti in 10 colonne.
    # (Su iris non si vedrebbe: con 3 classi le coppie sono 3, come le classi.)
    for shape in ["ovo", "ovr"]:
        m = SVC(kernel="rbf", C=1, gamma="auto", decision_function_shape=shape).fit(X18, y)
        print(f"decision_function_shape='{shape}': {m.decision_function(X18[:1]).shape[1]} colonne")
    print("-> SVC e' one-vs-one: il one-vs-rest del paper e' impreciso")


def svm_paper():
    print("\n===== SVM del paper: 3 validazioni x 2 preprocessing =====")
    schemes = {
        "stratificata, shuffle, seed 0": (StratifiedKFold(10, shuffle=True, random_state=0), None),
        "10-fold senza shuffle (paper)": (KFold(10), None),
        "per partecipante (GroupKFold)": (GroupKFold(10), groups),
    }
    rows = []
    for name, (cv, g) in schemes.items():
        for variant in ["globale", "nel fold"]:
            if variant == "globale":
                data, model = X18, make_svm()
            else:                                  # preprocessing dentro ogni fold
                data, model = X, make_pipeline(make_preprocessor(), make_svm())
            r = cross_validate(model, data, y, cv=cv, groups=g,
                               scoring=["accuracy", "f1_weighted"])
            rows.append({"validazione": name, "preprocessing": variant,
                         "accuracy": 100 * r["test_accuracy"].mean(),
                         "f1": 100 * r["test_f1_weighted"].mean(),
                         "per fold": np.round(100 * r["test_accuracy"], 1).tolist()})
    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))
    print("paper (Tab. 7, SVM) per fold: [85.2, 95.7, 96.5, 95.7, 87.0, 87.8, 87.0, 85.2, 97.4, 91.3]")
    table.to_csv(RESULTS / "svm.csv", index=False)


def esn_sec52():
    print("\n===== ESN, sez. 5.2, split 70/30 (paper: acc 92.08, F1 91.90) =====")
    # Verifiche su un'istanza (i test completi sono in esn.py: pytest esn.py)
    esn = ESN(n_channels=1, n_classes=10, seed=0, **ESN_SEC52)
    traj = esn.run_reservoir(U_seq[:3], return_all=True)
    print(f"raggio spettrale: voluto 1.3, misurato {np.max(np.abs(np.linalg.eigvals(esn.W))):.6f}")
    print(f"max |stato| = {np.abs(traj).max():.6f} (deve essere <= 1)")

    # Le tre codifiche dell'input dello scheletro
    rows = []
    for mode in MODES:
        U = encode_input(X18, mode)
        for seed in SEEDS:
            tr, te = split_70_30(seed)
            acc, f1 = train_ESN(U[tr], y[tr], U[te], y[te], ESN_SEC52, seed)
            rows.append({"codifica": mode, "seed": seed, "accuracy": acc, "f1": f1})
    table = pd.DataFrame(rows)
    print(table.groupby("codifica")[["accuracy", "f1"]].mean().round(2))
    table.to_csv(RESULTS / "esn_sec52.csv", index=False)


def esn_fig5():
    print("\n===== Fig. 5: accuracy al variare del raggio spettrale =====")
    rhos = [0.1, 0.3, 0.5, 0.7, 1.0, 1.12, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    paper = [24.5, 36.5, 42.5, 51.5, 72.5, 76.5, 79.5, 78, 75, 61, 34.5, 31]  # letti dal grafico
    table = pd.DataFrame({"rho": rhos, "paper (circa)": paper})
    for mode in MODES:
        U = encode_input(X18, mode)
        accs = []
        for rho in rhos:
            runs = []
            for seed in SEEDS:
                tr, te = split_70_30(seed)
                runs.append(train_ESN(U[tr], y[tr], U[te], y[te],
                                      {**ESN_FIG5, "spectral_radius": rho}, seed)[0])
            accs.append(round(np.mean(runs), 1))
        table[mode] = accs
    print(table.to_string(index=False))
    table.to_csv(RESULTS / "esn_fig5.csv", index=False)

    plt.figure(figsize=(6, 3.5))
    for mode in MODES:
        plt.plot(rhos, table[mode], "o-", label=mode)
    plt.plot(rhos, paper, "k--", label="paper (Fig. 5)")
    plt.xlabel("raggio spettrale"); plt.ylabel("accuracy (%)")
    plt.legend(); plt.tight_layout()
    plt.savefig(IMGS / "esn_fig5.png", dpi=120)


def hesnsvm():
    print("\n===== HESNSVM, Tab. 6, 10-fold senza shuffle (paper: 90.9 / 91.2 / 91.7) =====")
    rows = []
    for seed in SEEDS:
        for fold, (tr, te) in enumerate(KFold(10).split(X18), start=1):
            out = train_HESNSVM(U_seq[tr], y[tr], U_seq[te], y[te], ESN_TAB6, seed)
            out["SVM (feature)"] = train_SVM(X18[tr], y[tr], X18[te], y[te])
            for model, (acc, f1) in out.items():
                rows.append({"seed": seed, "fold": fold, "modello": model,
                             "accuracy": acc, "f1": f1})
        print(f"  seed {seed} completato")
    table = pd.DataFrame(rows)
    print(table.groupby("modello")[["accuracy", "f1"]].mean().round(2))
    table.to_csv(RESULTS / "hesnsvm.csv", index=False)


def export_bdesn():
    """Salva i dati nel formato .mat letto dal main.py di BDESN."""
    tr, te = split_70_30(0)
    out = ROOT / "bdesn" / "ACT.mat"
    out.parent.mkdir(exist_ok=True)
    scipy.io.savemat(out, {"X": U_seq[tr], "Y": y[tr].reshape(-1, 1) + 1,   # classi 1..10
                           "Xte": U_seq[te], "Yte": y[te].reshape(-1, 1) + 1})
    print(f"\nsalvato {out}  (X {U_seq[tr].shape}, Xte {U_seq[te].shape})")


if __name__ == "__main__":
    save_versions()
    if RUN_DATASET_CHECK: dataset_check()
    if RUN_SVM_IRIS: svm_iris()
    if RUN_SVM: svm_paper()
    if RUN_ESN: esn_sec52()
    if RUN_FIG5: esn_fig5()
    if RUN_HESNSVM: hesnsvm()
    if EXPORT_BDESN: export_bdesn()
