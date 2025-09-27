import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Union
from utils import hampel_on_residual
from statsmodels.tsa.stattools import acf


class ValidationPipeline:
    def __init__(
        self,
        model,
        metrics: Dict[str, Callable],
        mode: str = "tscv",  # "tscv" o "walkforward"
        params: Dict = None,
        seasonal_period: Union[int, str, None] = None,
        hampel_cfg: Dict = None
    ):
        """
        Pipeline de validación para series temporales con anomalías.
        
        Parameters
        ----------
        model : BaseAnomalyDetector
            Detector que implementa fit/predict/anomaly_score.
        metrics : dict
            Diccionario con nombre -> función de métrica.
        mode : str
            'tscv' (expanding window) o 'walkforward' (single-block update).
        params : dict
            Parámetros de splitting según el modo:
              - tscv: {L_train_min, L_val, G, S}
              - walkforward: {L_train0, L_blk}
        seasonal_period : int | "auto" | None
            - int: periodo estacional fijo (ej. 24, 168).
            - "auto": se estima automáticamente con autocorrelación.
            - None: no se aplica descomposición estacional.
        hampel_cfg : dict
            Configuración del filtro Hampel (window, sigma, causal, etc.).
        """
        self.model = model
        self.metrics = metrics
        self.mode = mode
        self.params = params or {}
        self.seasonal_period = seasonal_period
        self.hampel_cfg = hampel_cfg or {}

    # ---------------------
    # División temporal
    # ---------------------
    def _make_splits_tscv(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        L_train_min = self.params.get("L_train_min", 60)
        L_val = self.params.get("L_val", 30)
        G = self.params.get("G", 0)
        S = self.params.get("S", L_val)

        splits = []
        t_j = L_train_min - 1
        while t_j + G + L_val < T:
            train_idx = np.arange(0, t_j + 1)
            val_idx = np.arange(t_j + G + 1, t_j + G + L_val + 1)
            splits.append((train_idx, val_idx))
            t_j += S
        return splits

    def _make_splits_wf(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        L_train0 = self.params.get("L_train0", 60)
        L_blk = self.params.get("L_blk", 30)

        blocks = []
        start = 0
        blocks.append((0, L_train0 - 1))  # B0
        start = L_train0
        while start + L_blk <= T:
            blocks.append((start, start + L_blk - 1))
            start += L_blk

        splits = []
        for j in range(len(blocks) - 1):
            tr_idx = np.arange(blocks[j][0], blocks[j][1] + 1)
            va_idx = np.arange(blocks[j+1][0], blocks[j+1][1] + 1)
            splits.append((tr_idx, va_idx))
        return splits

    # ---------------------
    # Limpieza Hampel
    # ---------------------
    def _clean_block(self, y_block: pd.Series) -> pd.Series:
        seasonal_period = None  # valor por defecto

        if isinstance(self.seasonal_period, int) and self.seasonal_period >= 2:
            seasonal_period = int(self.seasonal_period)

        elif self.seasonal_period == "auto":
            est = int(self.estimate_seasonal_period(y_block))
            if est >= 2:
                seasonal_period = est
            else:
                seasonal_period = None  # ignoramos valores inválidos

        cleaned, _ = hampel_on_residual(
            y_block,
            seasonal_period=seasonal_period,
            return_mask=True,
            **self.hampel_cfg
        )
        return cleaned
    
    def estimate_seasonal_period(self, y, max_lag=200):
        acf_vals = acf(y, nlags=max_lag)
        # buscamos el primer pico "fuerte" después del lag=0
        peak_lag = np.argmax(acf_vals[1:]) + 1
        return peak_lag

    # ---------------------
    # Loop principal
    # ---------------------
    def validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        T = len(X)
        splits = (
            self._make_splits_tscv(T) if self.mode == "tscv" else self._make_splits_wf(T)
        )

        scores = {m: [] for m in self.metrics}

        for tr_idx, va_idx in splits:
            # Limpieza Hampel en el bloque de train
            X_train = X.iloc[tr_idx].copy()
            y_train = y.iloc[tr_idx].copy()
            X_train_clean = X_train.copy()
            for col in X_train.columns:
                X_train_clean[col] = self._clean_block(X_train[col])

            # Validación
            X_val, y_val = X.iloc[va_idx], y.iloc[va_idx]

            y_val = y_val.reset_index(drop=True).to_numpy().astype(int)

            # Preprocesado específico del modelo
            X_train_prep = self.model.preprocess(X_train_clean)
            X_val_prep = self.model.preprocess(X_val)

            # Entrenamiento + predicción
            self.model.fit(X_train_prep)
            y_pred = self.model.predict(X_val_prep)
            y_score = self.model.anomaly_score(X_val_prep)

            # Evaluación de métricas
            for name, metric in self.metrics.items():
                score = metric(y_val, y_pred, y_score)
                scores[name].append(score)

        return {m: np.nanmean(vals) for m, vals in scores.items()}

