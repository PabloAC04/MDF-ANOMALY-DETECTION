import cupy as cp
try:
    from cuml.decomposition import PCA as PCAcuml
    from cuml.preprocessing import StandardScaler as StandardScalerCuml
except ImportError:
    PCAcuml = None
    StandardScalerCuml = None
from sklearn.decomposition import PCA as PCAsk
from sklearn.preprocessing import StandardScaler as StandardScalerSk
import cudf
from .base import BaseAnomalyDetector  
import numpy as np

class PCAAnomalyDetectorGPU(BaseAnomalyDetector):
    def __init__(self, n_components=None, threshold=None):
        """
        n_components: número de componentes principales a retener.
                      Si None, se usa el criterio de máxima varianza explicada.
        threshold: percentil (ej. 0.95 => 95% de los errores por debajo).
        """
        self.n_components = int(n_components) if n_components is not None else None
        self.threshold = threshold
        self.pca = None
        self.scaler = StandardScalerCuml(with_mean=True, with_std=True)
        self._threshold_value = None

    def preprocess(self, X, retrain=True):
        """
        Convierte X a float32 y lo pasa a GPU (cupy).
        Si hay un scaler configurado, lo aplica en CPU antes de mover a GPU.
        """
        
        X = X.astype("float32")

        # aplicar scaler en GPU
        if retrain:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit(self, X):
        """
        Ajusta el modelo PCA en GPU y calcula el threshold.
        """
        self.pca = PCAcuml(n_components=self.n_components, svd_solver="full")
        self.pca.fit(X)

        errors = self._reconstruction_error(X)

        mu, sigma = cp.mean(errors), cp.std(errors)  # en GPU
        n_sigma = self.threshold if self.threshold is not None else 3.0 # default 3 sigma
        self._threshold_value = (mu + n_sigma * sigma).item() 

    def predict(self, X, y=None):
        """
        Devuelve etiquetas: 0 = normal, 1 = anómalo
        """
        errors = self._reconstruction_error(X)
        return cp.where(errors > self._threshold_value, 1, 0).get()

    def anomaly_score(self, X, y=None):
        """
        Devuelve el error de reconstrucción (mayor = más anómalo)
        """
        return self._reconstruction_error(X).get()

    def _reconstruction_error(self, X_proc):
        """
        Calcula el error de reconstrucción de cada muestra (en GPU).
        """
        X_projected = self.pca.inverse_transform(self.pca.transform(X_proc))
        errors = cp.mean((X_proc - X_projected) ** 2, axis=1)
        return errors.values

class PCAAnomalyDetectorCPU(BaseAnomalyDetector):
    def __init__(self, n_components=None, threshold=None):
        self.n_components = int(n_components) if n_components is not None else None
        self.threshold = threshold
        self.pca = None
        self.scaler = StandardScalerSk(with_mean=True, with_std=True)
        self._threshold_value = None

    def preprocess(self, X, retrain=True):
        X = X.astype("float32")
        if retrain:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def fit(self, X):
        self.pca = PCAsk(n_components=self.n_components, svd_solver="full")
        self.pca.fit(X)

        errors = self._reconstruction_error(X)
        
        mu, sigma = np.mean(errors), np.std(errors)  # en CPU
        n_sigma = self.threshold if self.threshold is not None else 3.0  # default 3 sigma
        self._threshold_value = mu + n_sigma * sigma

    def predict(self, X):
        errors = self._reconstruction_error(X)
        return (errors > self._threshold_value).astype(int)

    def anomaly_score(self, X):
        return self._reconstruction_error(X)

    def _reconstruction_error(self, X_proc):
        X_projected = self.pca.inverse_transform(self.pca.transform(X_proc))
        errors = np.mean((X_proc - X_projected) ** 2, axis=1)
        return errors

import numpy as np

def candidate_n_components(pca, threshold=0.95, min_candidates=6):
    """
    Genera un conjunto de valores candidatos para n_components en PCA
    usando 4 criterios clásicos y los expande con vecinos cercanos hasta
    alcanzar al menos 'min_candidates' valores distintos.

    Métodos usados:
      1. Varianza acumulada mínima (ej. ≥95%)
      2. Kaiser rule (autovalores > 1)
      3. Broken Stick Model
      4. Elbow method

    Parameters
    ----------
    pca : cuml.decomposition.PCA o sklearn.decomposition.PCA ya ajustado
        Modelo PCA entrenado con fit().
    threshold : float, opcional
        Nivel de varianza acumulada para el criterio 1. Por defecto 0.95.
    min_candidates : int, opcional
        Número mínimo de valores a devolver. Por defecto 6.

    Returns
    -------
    list of int
        Valores únicos de n_components sugeridos, ordenados.
    """

    # Extraer varianzas explicadas y autovalores en numpy
    expl_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    if hasattr(expl_var, "to_numpy"):  # cudf.Series
        expl_var = expl_var.to_numpy()
    elif hasattr(expl_var, "get"):     # cupy.ndarray
        expl_var = expl_var.get()

    if hasattr(eigenvalues, "to_numpy"):
        eigenvalues = eigenvalues.to_numpy()
    elif hasattr(eigenvalues, "get"):
        eigenvalues = eigenvalues.get()

    n = len(expl_var)

    # 1) Varianza acumulada mínima
    cum_var = np.cumsum(expl_var)
    n_var95 = np.argmax(cum_var >= threshold) + 1

    # 2) Kaiser rule
    n_kaiser = int(np.sum(eigenvalues > 1))

    # 3) Broken Stick
    broken_stick = np.array([np.sum(1.0/np.arange(k, n+1)) / n for k in range(1, n+1)])
    n_broken = int(np.sum(expl_var > broken_stick))

    # 4) Elbow method (geométrico)
    points = np.column_stack((np.arange(1, n+1), cum_var))
    start, end = points[0], points[-1]
    line_vec = end - start
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = np.abs(np.cross(points - start, line_vec_norm))
    n_elbow = int(np.argmax(distances) + 1)

    # Unir sin duplicados
    base_candidates = sorted(set([n_var95, n_kaiser, n_broken, n_elbow]))

    # Añadir vecinos hacia atrás (n-1, n-2, ...) hasta alcanzar min_candidates
    candidates = set(base_candidates)
    for val in base_candidates:
        for step in [1, 2]:  # vecinos por detrás
            if val - step >= 1 and len(candidates) < min_candidates:
                candidates.add(val - step)

    # Si aún faltan valores, añadir hacia adelante (por si el dataset es pequeño)
    for val in base_candidates:
        for step in [1, 2]:
            if val + step <= n and len(candidates) < min_candidates:
                candidates.add(val + step)

    return sorted(candidates)

