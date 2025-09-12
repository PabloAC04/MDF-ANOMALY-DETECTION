import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
from types import SimpleNamespace
import pandas as pd
from typing import Optional, Tuple, List, Literal
from statsmodels.tsa.statespace.varmax import VARMAX
from dataclasses import dataclass
from scipy.stats import chi2
from numpy.linalg import inv

CombineT = Literal["mahal", "l2"]

def _difference_df(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Aplica diferenciación d veces por columna."""
    if d <= 0:
        return df.copy()
    out = df.copy()
    for _ in range(d):
        out = out.diff()
    return out.dropna()

def _difference_with_history(X_eval: pd.DataFrame, hist: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    Diferencia X_eval usando 'hist' (últimas d filas del training original) para no perder borde.
    Devuelve X_eval ya diferenciado d veces y alineado fila a fila con X_eval.
    """
    if d <= 0:
        return X_eval.copy()
    assert len(hist) >= d, "Se requieren al menos d filas de historial para diferenciar."
    concat = pd.concat([hist, X_eval], axis=0)
    diffed = _difference_df(concat, d)
    return diffed.iloc[-len(X_eval):].copy()

@dataclass
class _FitState:
    model: VARMAX
    res: object                 # VARMAXResults
    Sigma: np.ndarray           # covarianza de innovaciones (k x k)
    Sigma_inv: np.ndarray       # inversa regularizada
    sigma_diag: np.ndarray      # desv. típicas por variable (para 'l2')

class VARIMADetector:
    """
    Detector de anomalías multivariante (VARIMA) mediante VARMA sobre serie diferenciada d veces.

    - order: (p, q) para la parte VARMA. (La 'I' se maneja con 'd' por diferenciación previa.)
    - d: orden de diferenciación común a todas las columnas (entero >= 0).
    - contamination:
        * None  -> umbral teórico chi2 (1 - alpha) con df = k
        * (0,1] -> umbral = cuantil 1 - contamination del score en training
    - alpha: nivel para el umbral teórico si contamination es None
    - combine:
        * "mahal" -> Mahalanobis D^2 = r^T Σ^{-1} r
        * "l2"    -> ||z||^2 con z = r / diag(Σ)^(1/2)
    - trend: "n" (sin constante) o "c" (con constante). Por defecto "n".
    """

    def __init__(self,
                 order: Tuple[int, int] = (1, 0),
                 d: int = 0,
                 contamination: Optional[float] = None,
                 alpha: float = 0.01,
                 combine: CombineT = "mahal",
                 trend: str = "n"):
        self.order = order
        self.d = int(d)
        self.contamination = contamination
        self.alpha = alpha
        self.combine = combine
        self.trend = trend

        # Estado aprendido
        self._fit: Optional[_FitState] = None
        self.columns_: Optional[List[str]] = None
        self.k_: Optional[int] = None
        self.threshold_: Optional[float] = None
        self._train_tail_: Optional[pd.DataFrame] = None  # últimas d filas del training original (para diferenciar eval)

    # ------------------ Ajuste ------------------
    def fit(self, X_train: pd.DataFrame):
        assert isinstance(X_train, pd.DataFrame), "X_train debe ser un DataFrame"
        self.columns_ = list(X_train.columns)
        self.k_ = len(self.columns_)
        assert self.k_ >= 1, "Se requiere al menos una variable"
        X_train = X_train.astype(float)

        # Guardar historial (últimas d filas del original) para diferenciar eval
        self._train_tail_ = X_train.tail(self.d) if self.d > 0 else X_train.tail(0)

        # Diferenciar training
        Xtr_diff = _difference_df(X_train, self.d)
        if len(Xtr_diff) < max(self.order[0], self.order[1]) + 5:
            raise ValueError("Training diferenciado demasiado corto para el orden VARMA especificado.")

        # Ajustar VARMA(p,q) (vía VARMAX sin exógenas)
        model = VARMAX(endog=Xtr_diff, order=self.order, trend=self.trend,
                       enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=500)

        # Innovaciones (residuos) en training diferenciado
        resid = res.resid.dropna()
        R = resid.to_numpy()  # (T', k)

        # Covarianza de innovaciones + regularización mínima
        Sigma = np.cov(R.T) if R.ndim == 2 and R.shape[0] > 1 else np.atleast_2d(np.var(R, axis=0))
        Sigma = np.atleast_2d(Sigma)
        eps = 1e-8
        Sigma = Sigma + eps * np.eye(self.k_)
        Sigma_inv = inv(Sigma)
        sigma_diag = np.sqrt(np.clip(np.diag(Sigma), eps, None))

        self._fit = _FitState(model=model, res=res, Sigma=Sigma, Sigma_inv=Sigma_inv, sigma_diag=sigma_diag)

        # Scores en training (D^2 o ||z||^2) para fijar umbral si hay contamination
        scores_train = self._scores_from_residuals(R)

        if self.contamination is not None:
            q = float(1.0 - self.contamination)
            q = min(max(q, 0.0), 1.0)
            self.threshold_ = float(np.quantile(scores_train, q))
        else:
            # Umbral teórico chi-cuadrado con k g.l. (D^2 o ||z||^2 ~ chi2_k)
            self.threshold_ = float(chi2.ppf(1.0 - self.alpha, df=self.k_))
        return self

    # ------------------ Puntuación / Predicción ------------------
    def anomaly_score(self, X_eval: pd.DataFrame) -> np.ndarray:
        """
        Score por instante usando rolling one-step-ahead en la serie DIFERENCIADA.
        Devuelve D^2 (Mahalanobis) o ||z||^2 por tiempo.
        """
        self._check_fitted()
        assert list(X_eval.columns) == self.columns_, "Columnas de X_eval deben coincidir con training"
        X_eval = X_eval.astype(float)

        # Diferenciar eval con historial del training
        Xev_diff = _difference_with_history(X_eval, self._train_tail_, self.d)

        # Rolling: pronóstico one-step en espacio diferenciado
        res_roll = self._fit.res
        scores: List[float] = []

        for _, y_t_diff in Xev_diff.iterrows():
            # Predicción 1 paso (en espacio diferenciado)
            yhat = res_roll.forecast(steps=1).iloc[0].values  # (k,)
            r = (y_t_diff.values - yhat).astype(float)        # innovación
            scores.append(self._score_vector(r))
            # Actualizar filtro con observación (diferenciada)
            res_roll = res_roll.append(y_t_diff.to_frame().T)

        return np.asarray(scores)

    def predict(self, X_eval: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        tau = float(self.threshold_ if threshold is None else threshold)
        s = self.anomaly_score(X_eval)
        return (s >= tau).astype(int)

    # ------------------ Utilidades internas ------------------
    def _scores_from_residuals(self, R: np.ndarray) -> np.ndarray:
        if self.combine == "mahal":
            # D^2 = r^T Σ^-1 r
            return np.einsum("ti,ij,tj->t", R, self._fit.Sigma_inv, R)
        else:
            # ||z||^2 con z = r / sigma_diag
            Z = R / (self._fit.sigma_diag + 1e-12)
            return np.sum(Z**2, axis=1)

    def _score_vector(self, r: np.ndarray) -> float:
        if self.combine == "mahal":
            return float(r @ self._fit.Sigma_inv @ r)
        else:
            z = r / (self._fit.sigma_diag + 1e-12)
            return float(np.sum(z**2))

    def _check_fitted(self):
        if self._fit is None or self.threshold_ is None or self.columns_ is None:
            raise RuntimeError("Debes llamar a fit(X_train) antes de puntuar o predecir.")