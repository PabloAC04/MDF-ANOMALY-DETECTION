import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        self.scaler = None
        self._threshold_value = None

    def preprocess(self, X):
        """
        Estandariza los datos (media=0, varianza=1) para que
        ninguna señal domine en el PCA.
        """
        X = np.asarray(X)
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def fit(self, X):
        X_proc = self.preprocess(X)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_proc)

        errors = self._reconstruction_error(X_proc)
        q = self.threshold if self.threshold is not None else 0.997  # como 3*std aprox
        self._threshold_value = np.quantile(errors, q)

    def predict(self, X):
        """
        Devuelve etiquetas: 0 = normal, 1 = anómalo
        """
        X_proc = self.preprocess(X)
        errors = self._reconstruction_error(X_proc)
        return np.where(errors > self._threshold_value, 1, 0)

    def anomaly_score(self, X):
        """
        Devuelve el error de reconstrucción (mayor = más anómalo)
        """
        X_proc = self.preprocess(X)
        return self._reconstruction_error(X_proc)

    def _reconstruction_error(self, X_proc):
        """
        Calcula el error de reconstrucción de cada muestra (ya preprocesada).
        """
        X_projected = self.pca.inverse_transform(self.pca.transform(X_proc))
        errors = np.mean((X_proc - X_projected) ** 2, axis=1)
        return np.asarray(errors)
