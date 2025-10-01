import cupy as cp
from cuml.decomposition import PCA
from cuml.preprocessing import StandardScaler
import cudf
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
        self.scaler = StandardScaler(with_mean=True, with_std=True)
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
        return errors.values
