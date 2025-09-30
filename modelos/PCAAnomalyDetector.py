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
        self.n_components = int(n_components) if n_components is not None else None
        self.threshold = threshold
        self.pca = None
        self.scaler = None
        self._threshold_value = None

    def preprocess(self, X, retrain=True):
        """
        Convierte X a float32 y lo pasa a GPU (cupy).
        Si hay un scaler configurado, lo aplica en CPU antes de mover a GPU.
        """
        import numpy as np  # solo para CPU scaler
        X = np.asarray(X, dtype=np.float32)

        if self.scaler:
            if retrain:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)

        return cp.asarray(X)  # mover a GPU

    def fit(self, X):
        """
        Ajusta el modelo PCA en GPU y calcula el threshold.
        """
        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        self.pca.fit(X)

        errors = self._reconstruction_error(X)
        q = self.threshold if self.threshold is not None else 0.997  # ~3*std
        self._threshold_value = cp.quantile(errors, q).item()  # pasar a float CPU

    def predict(self, X):
        """
        Devuelve etiquetas: 0 = normal, 1 = anómalo
        """
        errors = self._reconstruction_error(X)
        return cp.where(errors > self._threshold_value, 1, 0).get()

    def anomaly_score(self, X):
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
        return errors
