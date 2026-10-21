"""
Scheletro per l'Echo State Network in NumPy -- COMPLETATO.

Firme delle funzioni e della classe identiche allo scheletro consegnato.
Aggiunte (tutte facoltative, non cambiano l'interfaccia):
    - run_reservoir(U, x0=None, return_all=False): x0 e return_all servono
      al test della echo state property e al grafico delle traiettorie;
    - fit_states / decision_from_states: lo stesso readout, ma partendo da
      stati gia' calcolati (HESNSVM usa gli stessi stati per ESN e SVM).

Notazione del paper (Tabella 5):
    Nr  dimensione del reservoir
    beta  percentuale di connessioni non nulle nel reservoir
    rho   raggio spettrale
    omega scala dei pesi di input
    noise rumore nell'aggiornamento dello stato (simbolo "£" nel paper)
    lam   coefficiente di regolarizzazione ridge (lambda)

Esecuzione (dalla cartella code/):
    pytest -v esn.py     i 5 test di sanita'
    python esn.py        esempio minimo: sez. 5.2, split 70/30, 3 seed
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Rappresentazione dell'input
# ---------------------------------------------------------------------------

def encode_input(X: np.ndarray, mode: str, n_repeat: int = 10) -> np.ndarray:
    """
    Trasforma le feature (n_campioni, n_feature) in una sequenza
    (n_campioni, n_passi, n_canali) da dare al reservoir.

    Il paper non dice come un vettore di feature entri nel reservoir.
    Modalita' da confrontare:
        "single_step"          tutte le feature in un solo passo  -> (N, 1, 18)
        "feature_as_sequence"  una feature per passo               -> (N, 18, 1)
        "repeat"               lo stesso vettore ripetuto per T passi
                                                          -> (N, n_repeat, 18)

    Cosa si osserva su rho (Fig. 5 del paper):
      * "single_step": con stato iniziale nullo il termine W x(t-1) vale
        sempre zero, quindi rho non ha NESSUN effetto (accuracy piatta ~92%).
      * "feature_as_sequence": accuracy a campana in rho, 28% -> 79% -> 31%,
        quasi identica al paper (25% -> 80% -> 31%).
      * "repeat": quasi piatta (~92%) fino a rho ~1.7, poi cala (83% a
        rho = 3): l'input costante "trattiene" il reservoir e rho conta poco.
    Il paper quindi dava le feature una per passo: e' la modalita' scelta.
    """
    X = np.asarray(X, dtype=float)
    if mode == "single_step":
        return X[:, None, :]
    if mode == "feature_as_sequence":
        return X[:, :, None]
    if mode == "repeat":
        return np.repeat(X[:, None, :], n_repeat, axis=1)
    raise ValueError(f"modalita' sconosciuta: {mode}")


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------

def make_reservoir_matrix(n_reservoir: int, sparsity: float,
                          spectral_radius: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Matrice ricorrente casuale e sparsa, con raggio spettrale rho.

    Ordine delle operazioni:
      1. pesi uniformi in [-0.5, 0.5] (come il codice di riferimento)
      2. maschera di sparsita': resta una frazione `sparsity` (beta) dei pesi
      3. solo ora calcolo il raggio spettrale e riscalo a rho
    Per Nr fino a qualche migliaio basta np.linalg.eigvals (Nr=1500: pochi
    secondi); per reservoir piu' grandi si userebbe
    scipy.sparse.linalg.eigs(W, k=1, which='LM').
    """
    W = rng.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir))
    W = W * (rng.random((n_reservoir, n_reservoir)) < sparsity)
    rho_now = np.max(np.abs(np.linalg.eigvals(W)))
    return W * (spectral_radius / rho_now)


class ESN:
    def __init__(self, n_channels: int, n_reservoir: int, n_classes: int,
                 sparsity: float, spectral_radius: float,
                 input_scaling: float, noise: float = 0.0,
                 ridge: float = 1.0, seed: int = 0):
        self.n_channels = n_channels
        self.n_reservoir = n_reservoir
        self.n_classes = n_classes
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.noise = noise
        self.ridge = ridge
        self.seed = seed                        # salvato con i risultati
        self.rng = np.random.default_rng(seed)  # un seed per ogni run, da salvare

        # W_in densa, uniforme in [-1, 1] e scalata da omega (come il codice
        # di riferimento; con pesi +-1 l'accuracy scendeva a ~83%)
        self.W_in = input_scaling * self.rng.uniform(
            -1.0, 1.0, size=(n_reservoir, n_channels))
        self.W = make_reservoir_matrix(n_reservoir, sparsity,
                                       spectral_radius, self.rng)
        self.W_out = None   # (n_reservoir, n_classes), calcolato da fit
        self.b_out = None   # (n_classes,) intercetta del readout

    def run_reservoir(self, U: np.ndarray, x0: np.ndarray = None,
                      return_all: bool = False) -> np.ndarray:
        """
        U: (n_campioni, n_passi, n_canali).
        Restituisce un vettore di stato per campione: (n_campioni, n_reservoir).
        Con return_all=True restituisce la traiettoria (N, n_passi, Nr).

        Un solo ciclo, sui passi temporali: i campioni si trattano insieme
        con operazioni su matrici (la riga i di x e' lo stato del campione i).

        Decisioni (come il codice di riferimento usato dal paper):
            - al readout va l'ULTIMO stato
            - il rumore, uniforme in [0, noise), entra DENTRO la tanh
            - nessun bias nell'input
            - stato iniziale sempre a zero (x0 serve solo ai test)
        """
        n_samples, n_steps, _ = U.shape
        x = np.zeros((n_samples, self.n_reservoir)) if x0 is None else x0
        history = []
        for t in range(n_steps):
            pre = U[:, t, :] @ self.W_in.T + x @ self.W.T
            if self.noise > 0:
                pre = pre + self.noise * self.rng.random(pre.shape)
            x = np.tanh(pre)
            history.append(x)
        return np.stack(history, axis=1) if return_all else x

    def fit(self, U: np.ndarray, y: np.ndarray) -> "ESN":
        """
        Calcola gli stati, poi il readout con regressione ridge.
        y: etichette intere (n_campioni,). Il target e' one-hot.
        """
        return self.fit_states(self.run_reservoir(U), y)

    def fit_states(self, S: np.ndarray, y: np.ndarray) -> "ESN":
        """
        Readout ridge a partire dagli stati S (N, Nr).
        Minimizza ||S W + b - Y||^2 + lam ||W||^2 con b non regolarizzato,
        come sklearn.linear_model.Ridge. Per b: centro S e Y, risolvo senza b,
        poi b = media(Y) - media(S) W.
        Niente inversa, np.linalg.solve sul sistema piu' piccolo:
            primale (N >= Nr):  (Sc^T Sc + lam I) W = Sc^T Yc        Nr x Nr
            duale   (N <  Nr):  W = Sc^T (Sc Sc^T + lam I)^-1 Yc     N x N
        """
        Y = np.eye(self.n_classes)[y]
        s_mean, y_mean = S.mean(axis=0), Y.mean(axis=0)
        Sc, Yc = S - s_mean, Y - y_mean
        n, nr = Sc.shape
        if n >= nr:
            self.W_out = np.linalg.solve(Sc.T @ Sc + self.ridge * np.eye(nr), Sc.T @ Yc)
        else:
            self.W_out = Sc.T @ np.linalg.solve(Sc @ Sc.T + self.ridge * np.eye(n), Yc)
        self.b_out = y_mean - s_mean @ self.W_out
        return self

    def decision_from_states(self, S: np.ndarray) -> np.ndarray:
        return S @ self.W_out + self.b_out

    def decision_function(self, U: np.ndarray) -> np.ndarray:
        """Uscite grezze del readout (n_campioni, n_classi). Servono per HESNSVM."""
        return self.decision_from_states(self.run_reservoir(U))

    def predict(self, U: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_function(U), axis=1)


# ---------------------------------------------------------------------------
# Test di sanita' (da completare e far passare prima di fare esperimenti)
# ---------------------------------------------------------------------------

# Parametri della sezione 5.2 del paper
P52 = dict(n_reservoir=700, sparsity=0.33, spectral_radius=1.3,
           input_scaling=1.7, noise=1e-5, ridge=15.12)


def _paper_data():
    """Dati preprocessati e codificati una feature per passo."""
    from data_utils import load_dataset, preprocess_all
    X, y, _, _ = load_dataset()
    return encode_input(preprocess_all(X, y), "feature_as_sequence"), y


def test_spectral_radius():
    """Il raggio spettrale della matrice ricorrente e' davvero rho."""
    rng = np.random.default_rng(0)
    for rho in [0.5, 1.0, 1.3, 2.0]:
        W = make_reservoir_matrix(300, 0.33, rho, rng)
        assert abs(np.max(np.abs(np.linalg.eigvals(W))) - rho) < 1e-8
        assert abs(np.mean(W != 0) - 0.33) < 0.02        # sparsita' ~ beta


def test_states_bounded():
    """Gli stati restano limitati. Fai anche un grafico di qualche traiettoria."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U, _ = _paper_data()
    esn = ESN(n_channels=1, n_classes=10, **P52)
    traj = esn.run_reservoir(U[:3], return_all=True)          # (3, 18, 700)
    big = esn.run_reservoir(np.random.default_rng(1).normal(0, 5, (50, 100, 1)))
    assert np.all(np.isfinite(traj)) and np.abs(traj).max() <= 1
    assert np.abs(big).max() <= 1                    # anche con input enormi

    plt.figure(figsize=(7, 3))
    plt.plot(traj[0, :, :8])
    plt.xlabel("passo (= indice della feature)"); plt.ylabel("stato")
    plt.title("8 neuroni del reservoir, campione 0"); plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "imgs"
    out.mkdir(exist_ok=True)
    plt.savefig(out / "esn_trajectories.png", dpi=120)


def test_echo_state_property():
    """
    Due stati iniziali diversi, stesso input: le traiettorie devono
    convergere. Con rho > 1 non e' garantito dal solo raggio spettrale,
    quindi va verificato.
    """
    rng = np.random.default_rng(2)
    U = rng.normal(size=(5, 200, 1))
    esn = ESN(n_channels=1, n_classes=10, **{**P52, "noise": 0.0})
    xa = esn.run_reservoir(U, x0=rng.uniform(-1, 1, (5, 700)), return_all=True)
    xb = esn.run_reservoir(U, x0=rng.uniform(-1, 1, (5, 700)), return_all=True)
    dist = np.linalg.norm(xa - xb, axis=2)                    # (campioni, passi)
    assert np.all(dist[:, -1] < 1e-6 * dist[:, 0])


def test_ridge_matches_sklearn():
    """Il tuo readout coincide con sklearn.linear_model.Ridge sugli stessi stati."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(3)
    for n in [400, 50]:                     # n > Nr: primale, n < Nr: duale
        S, y = rng.normal(size=(n, 100)), rng.integers(0, 10, n)
        esn = ESN(n_channels=1, n_reservoir=100, n_classes=10, sparsity=0.3,
                  spectral_radius=1.0, input_scaling=1.0, ridge=2.5)
        esn.fit_states(S, y)
        reference = Ridge(alpha=2.5).fit(S, np.eye(10)[y]).predict(S)
        assert np.allclose(esn.decision_from_states(S), reference, atol=1e-8)


def test_recurrence_matters():
    """
    Confronta con una versione senza ricorrenza (W = 0). Se la tua ESN non
    la batte, la ricorrenza non sta facendo nulla.
    """
    from sklearn.model_selection import train_test_split
    U, y = _paper_data()
    Utr, Ute, ytr, yte = train_test_split(U, y, test_size=0.3, stratify=y,
                                          random_state=0)
    acc = {}
    for name in ["con W", "W = 0"]:
        esn = ESN(n_channels=1, n_classes=10, **P52)
        if name == "W = 0":
            esn.W[:] = 0.0
        acc[name] = np.mean(esn.fit(Utr, ytr).predict(Ute) == yte)
    print(acc)                                       # circa 0.91 contro 0.33
    assert acc["con W"] > acc["W = 0"] + 0.2


if __name__ == "__main__":
    # Esempio minimo.
    from sklearn.model_selection import train_test_split

    # 1. carica i dati preprocessati   2. encode_input
    U, y = _paper_data()
    # 3. split 70/30, ESN con i parametri della sezione 5.2 del paper
    # 4. accuracy su test, tre seed diversi
    accs = []
    for seed in [0, 1, 2]:
        Utr, Ute, ytr, yte = train_test_split(U, y, test_size=0.3, stratify=y,
                                              random_state=seed)
        esn = ESN(n_channels=1, n_classes=10, seed=seed, **P52).fit(Utr, ytr)
        accs.append(100 * np.mean(esn.predict(Ute) == yte))
        print(f"seed {seed}: accuracy {accs[-1]:.2f}%")
    print(f"media {np.mean(accs):.2f}%  (paper 92.08%)")
