import numpy as np
from sklearn.decomposition import PCA
from .base import BaseAnomalyDetector  

class PCAAnomalyDetector(BaseAnomalyDetector):
    def __init__(self, n_components=None, threshold=None):
        """
        n_components: número de componentes principales a retener.
                      Si None, se usa el criterio de máxima varianza explicada.
        threshold: percentil (ej. 0.95 => 95% de los errores por debajo).
        """
        self.n_components = n_components
        self.threshold = threshold
        self.pca = None
        self.mean_ = None
        self._threshold_value = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

        errors = self._reconstruction_error(X)
        q = self.threshold if self.threshold is not None else 0.997 # como 3*std en una normal
        self._threshold_value = np.quantile(errors, q)

    def predict(self, X):
        """
        Devuelve etiquetas: 1 = normal, -1 = anómalo
        """
        errors = self._reconstruction_error(X)
        return np.where(errors > self._threshold_value, 1, 0)

    def anomaly_score(self, X):
        """
        Devuelve el error de reconstrucción (mayor = más anómalo)
        """
        return self._reconstruction_error(X)

    def _reconstruction_error(self, X):
        """
        Calcula el error de reconstrucción de cada muestra.
        """
        X_centered = X - self.mean_
        X_projected = self.pca.inverse_transform(self.pca.transform(X_centered))
        errors = np.mean((X_centered - X_projected) ** 2, axis=1)
        return np.asarray(errors)
