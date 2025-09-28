import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from modelos.base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    def __init__(self, nu=0.05, kernel="rbf", gamma="scale", use_scaler=True):
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
            Si True, aplica normalización con StandardScaler.
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.use_scaler = use_scaler

        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.scaler = StandardScaler() if self.use_scaler else None
        self.is_fitted = False

    def preprocess(self, X):
        """Aplica normalización si se ha configurado el scaler."""
        X = np.asarray(X)
        if self.scaler is not None:
            if not self.is_fitted:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X

    def fit(self, X):
        """Entrena el modelo One-Class SVM sobre los datos X."""
        X_proc = self.preprocess(X)
        self.model.fit(X_proc)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predice etiquetas para los datos X.
        Retorna 1 para normal, -1 para anómalo.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de predecir.")
        X_proc = self.preprocess(X)
        y_pred = self.model.predict(X_proc)

        return np.where(y_pred == -1, 1, 0) # Convertir etiquetas: 1 -> 0 (normal), -1 -> 1 (anómalo)

    def anomaly_score(self, X):
        """
        Devuelve el score de anomalía (distancia a la frontera).
        Valores negativos indican mayor probabilidad de anomalía.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de calcular scores.")
        X_proc = self.preprocess(X)
        scores = self.model.decision_function(X_proc)

        return -scores
