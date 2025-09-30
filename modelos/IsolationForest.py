import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from modelos.datasets_to_parquet import load_project_parquets
from modelos.base import BaseAnomalyDetector

def harmonic_number(n):
    if n <= 1:
        return 0
    return np.log(n) + 0.5772156649

def c_factor(n):
    if n <= 1:
        return 0
    return 2 * harmonic_number(n - 1) - (2 * (n - 1) / n)

class iTree:
    def __init__(self, data, current_height=0, max_height=np.inf, split_selection_random=True):
        self.n_samples, self.n_features = data.shape
        self.left = None
        self.right = None
        self.split_attr = None
        self.split_value = None
        self.size = self.n_samples
        self.is_leaf = False

        if current_height >= max_height or self.n_samples <= 1:
            self.is_leaf = True
        else:
            self.split_attr = np.random.randint(self.n_features)
            min_val = np.min(data[:, self.split_attr])
            max_val = np.max(data[:, self.split_attr])

            if min_val == max_val:
                self.is_leaf = True
                return

            if split_selection_random:
                # Selección aleatoria del valor de división
                self.split_value = np.random.uniform(min_val, max_val)
            else:
                # Selección del valor de división como un valor aleatorio de la característica
                feat_values = data[:, self.split_attr]
                self.split_value = np.random.choice(feat_values)


            left_mask = data[:, self.split_attr] < self.split_value
            self.left = iTree(data[left_mask], current_height + 1, max_height, split_selection_random=split_selection_random)
            self.right = iTree(data[~left_mask], current_height + 1, max_height, split_selection_random=split_selection_random)

    def path_length(self, x, current_height=0):
        if self.is_leaf:
            return current_height + c_factor(self.size)
        if x[self.split_attr] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)

# ------------------------------
# Bosque de aislamiento
# ------------------------------
import numpy as np

class IsolationForest(BaseAnomalyDetector):
    """
    Isolation Forest clásico con una única política de umbral:
        - Si contamination is None  -> umbral fijo tau = 0.5
        - Si contamination in (0,1] -> umbral aprendido como quantile_{1-cont}(scores_train)
    Además, se puede elegir entre dos políticas de selección del punto de división en cada nodo:
        - split_selection_random=True: selección aleatoria uniforme entre min y max de la característica.
        - split_selection_random=False: selección aleatoria entre los valores presentes en la muestra.
    """
    def __init__(self,
                 n_trees=100,
                 sample_size=256,
                 split_selection_random=True,
                 contamination=None):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.split_selection_random = split_selection_random
        self.contamination = contamination  # None => tau=0.5 ; (0,1] => cuantil
        self.trees = []
        self.threshold_ = None  # se fija en fit()

    def fit(self, X):
        X = np.asarray(X)
        self.trees = []
        n = len(X)
        height_limit = int(np.ceil(np.log2(max(self.sample_size, 2))))
        for _ in range(self.n_trees):
            if self.sample_size < n:
                idx = np.random.choice(n, self.sample_size, replace=False)
            else:
                idx = np.random.choice(n, self.sample_size, replace=True)
            sample = X[idx]
            tree = iTree(sample, current_height=0, max_height=height_limit,
                         split_selection_random=self.split_selection_random)
            self.trees.append(tree)

        # --- Fijar umbral según contamination ---
        if self.contamination is None:
            # Frontera teórica clásica del score de iForest
            self.threshold_ = 0.5
        else:
            # Cuantil 1 - contamination de los scores sobre TRAIN
            scores_train = self.anomaly_score(X)
            q = float(1.0 - self.contamination)
            q = min(max(q, 0.0), 1.0)  # clamp por seguridad
            self.threshold_ = np.quantile(scores_train, q)
        return self

    def anomaly_score(self, X):
        X = np.asarray(X)
        scores = np.zeros(len(X), dtype=float)
        denom = c_factor(self.sample_size)
        if denom <= 0:
            denom = 1.0  # seguridad numérica
        for i, x in enumerate(X):
            path_lengths = np.array([t.path_length(x) for t in self.trees], dtype=float)
            E_h = np.mean(path_lengths)
            scores[i] = 2 ** (-E_h / denom)
        return scores

    def predict(self, X, threshold=None):
        """
        Si 'threshold' se pasa aquí, tiene prioridad para pruebas puntuales.
        En otro caso se usa self.threshold_ fijado en fit().
        """
        X = np.asarray(X)
        tau = threshold if threshold is not None else self.threshold_
        return self.anomaly_score(X) >= tau

