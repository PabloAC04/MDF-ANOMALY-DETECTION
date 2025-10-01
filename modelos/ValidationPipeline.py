import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Union
from utils import hampel_gpu
from statsmodels.tsa.stattools import acf
import cudf
import cupy as cp
from cupyx.scipy.ndimage import median_filter

import warnings


class ValidationPipeline:
    def __init__(
        self,
        model,
        metrics: Dict[str, Callable],
        mode: str = "tscv",  # "tscv" o "walkforward"
        params: Dict = None,
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
        self.hampel_cfg = hampel_cfg or {}
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

        # ---------------------
    # División temporal
    # ---------------------
    def _make_splits_tscv(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Time Series Cross-Validation (ventana expansiva)
        - P_train: porcentaje inicial para entrenamiento
        - num_windows: número de ventanas de validación
        """
        P_train = self.params.get("P_train", 0.5)
        num_windows = self.params.get("num_windows", 10)

        # tamaño inicial de train
        L_train = int(P_train * T)
        if L_train <= 0 or L_train >= T:
            return []

        # tamaño de cada bloque de validación
        L_val = (T - L_train) // num_windows
        if L_val <= 0:
            return []

        splits = []
        start_val = L_train
        for w in range(num_windows):
            end_val = start_val + L_val
            if end_val > T:
                break
            train_idx = np.arange(0, start_val)  # ventana expansiva: acumula todo hasta inicio de val
            val_idx = np.arange(start_val, end_val)
            splits.append((train_idx, val_idx))
            start_val = end_val
        return splits


    def _make_splits_wf(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Walkforward validation (bloques sucesivos)
        - P_train: porcentaje inicial para entrenamiento
        - num_windows: número de ventanas de validación
        """
        P_train = self.params.get("P_train", 0.5)
        num_windows = self.params.get("num_windows", 10)

        # tamaño inicial de train
        L_train0 = int(P_train * T)
        if L_train0 <= 0 or L_train0 >= T:
            return []

        # tamaño de cada bloque de validación
        L_blk = (T - L_train0) // num_windows
        if L_blk <= 0:
            return []

        splits = []
        start = L_train0
        for w in range(num_windows):
            end = start + L_blk
            if end > T:
                break
            tr_idx = np.arange(0, start)    # siempre desde el inicio hasta antes del bloque
            va_idx = np.arange(start, end)  # siguiente bloque
            splits.append((tr_idx, va_idx))
            start = end
        return splits


    # ---------------------
    # Limpieza Hampel
    # ---------------------
    def _clean_block(self, y_block: cudf.Series) -> pd.Series:
        y_cp = y_block.values  # cupy array en GPU
        cleaned, _ = hampel_gpu(y_cp, window=self.hampel_cfg.get("window", 25),
                                    sigma=self.hampel_cfg.get("sigma", 5.0))
        return cudf.Series(cleaned, index=y_block.index)
    
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

            y_val = cp.asarray(y_val.values).astype(cp.int32)

            # Preprocesado específico del modelo
            X_train_prep = self.model.preprocess(X_train_clean, retrain=True)
            X_val_prep = self.model.preprocess(X_val, retrain=False)

            # Entrenamiento + predicción
            self.model.fit(X_train_prep)
            y_pred = self.model.predict(X_val_prep)
            y_score = self.model.anomaly_score(X_val_prep)

            # Evaluación de métricas
            for name, metric in self.metrics.items():
                score = metric(
                    cp.asnumpy(y_val),   # CPU
                    cp.asnumpy(y_pred),
                    cp.asnumpy(y_score)
                )
                scores[name].append(score)

        return {m: np.nanmean(vals) for m, vals in scores.items()}

