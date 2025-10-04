import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler

from modelos.base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    def __init__(self, nu=0.05, kernel="rbf", gamma="scale"):
        """
        Detector de anomalías basado en One-Class SVM.

        Parámetros
        ----------
        nu : float, opcional (default=0.05)
            Fracción de outliers esperados en los datos.
        kernel : str, opcional (default="rbf")
            Tipo de kernel a utilizar ("linear", "rbf", "poly", "sigmoid").
        gamma : {"scale", "auto"} o float, opcional (default="scale")
            Parámetro del kernel RBF/polinómico/sigmoide.
        use_scaler : bool, opcional (default=True)
            Si True, aplica normalización con RobustScaler.
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.is_fitted = False

        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.scaler = RobustScaler()

    def preprocess(self, X, retrain=True):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler:
            if retrain:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X

    def fit(self, X):
        """Entrena el modelo One-Class SVM sobre los datos X."""
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X, y=None):
        """
        Predice etiquetas para los datos X.
        Retorna 1 para normal, -1 para anómalo.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de predecir.")
        y_pred = self.model.predict(X)

        return np.where(y_pred == -1, 1, 0) # Convertir etiquetas: 1 -> 0 (normal), -1 -> 1 (anómalo)

    def anomaly_score(self, X, y=None):
        """
        Devuelve el score de anomalía (distancia a la frontera).
        Valores negativos indican mayor probabilidad de anomalía.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de calcular scores.")
        scores = self.model.decision_function(X)

        return -scores