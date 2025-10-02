
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple
from utils import hampel_gpu, hampel_cpu
from statsmodels.tsa.stattools import acf
import cudf
import cupy as cp

import warnings


from abc import ABC, abstractmethod
import numpy as np

class ValidationPipelineBase(ABC):
    def __init__(self, model, metrics, mode="tscv", params=None, hampel_cfg=None):
        self.model = model
        self.metrics = metrics
        self.mode = mode
        self.params = params or {}
        self.hampel_cfg = hampel_cfg or {}

    @abstractmethod
    def _clean_block(self, y_block):
        """Implementación específica (CPU/GPU)."""
        pass

    @abstractmethod
    def _to_numpy(self, arr):
        """Conversión a numpy para métricas."""
        pass

    def validate(self, X, y):
        T = len(X)
        if self.mode == "tscv":
            splits = make_splits_tscv(T, self.params.get("P_train", 0.5), self.params.get("num_windows", 10))
        else:
            splits = make_splits_wf(T, self.params.get("P_train", 0.5), self.params.get("num_windows", 10))

        scores = {m: [] for m in self.metrics}

        for tr_idx, va_idx in splits:
            X_train = X.iloc[tr_idx].copy()
            X_train_clean = X_train.copy()
            for col in X_train.columns:
                X_train_clean[col] = self._clean_block(X_train[col])

            X_val, y_val = X.iloc[va_idx], y.iloc[va_idx]

            # Preprocesado específico del modelo
            X_train_prep = self.model.preprocess(X_train_clean, retrain=True)
            X_val_prep = self.model.preprocess(X_val, retrain=False)

            # Fit + predict
            self.model.fit(X_train_prep)
            y_pred = self.model.predict(X_val_prep, y_val)
            y_score = self.model.anomaly_score(X_val_prep, y_val)

            if isinstance(y_pred, tuple):
                y_pred, y_val = y_pred
            if isinstance(y_score, tuple):
                y_score, y_val = y_score

            # Evaluación de métricas
            for name, metric in self.metrics.items():
                score = metric(
                    self._to_numpy(y_val),
                    self._to_numpy(y_pred),
                    self._to_numpy(y_score),
                )
                scores[name].append(score)

        return {m: np.nanmean(vals) for m, vals in scores.items()}


# CPU
class ValidationPipelineCPU(ValidationPipelineBase):
    def _clean_block(self, y_block: pd.Series):
        y_np = y_block.to_numpy(dtype=float)
        cleaned, _ = hampel_cpu(y_np,
                                window=self.hampel_cfg.get("window", 25),
                                sigma=self.hampel_cfg.get("sigma", 5.0))
        return cleaned

    def _to_numpy(self, arr):
        return np.asarray(arr)  # ya es np en CPU


# GPU
import cupy as cp

class ValidationPipelineGPU(ValidationPipelineBase):
    def _clean_block(self, y_block: cudf.Series):
        y_block = y_block.ffill().bfill().fillna(0)
        y_cp = y_block.values
        cleaned, _ = hampel_gpu(y_cp,
                                window=self.hampel_cfg.get("window", 25),
                                sigma=self.hampel_cfg.get("sigma", 5.0))
        return cleaned

    def _to_numpy(self, arr):
        return cp.asnumpy(arr)  # pasa de GPU → CPU para métricas


def make_splits_tscv(T: int, P_train: float = 0.5, num_windows: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Time Series Cross-Validation (ventana deslizante / rolling window)."""
    L_train = int(P_train * T)
    if L_train <= 0 or L_train >= T:
        return []

    L_val = (T - L_train) // num_windows
    if L_val <= 0:
        return []

    splits = []
    start_val = L_train
    for _ in range(num_windows):
        end_val = start_val + L_val
        if end_val > T:
            break
        # --- Rolling window: usar solo los últimos L_train ---
        train_idx = np.arange(start_val - L_train, start_val)
        val_idx = np.arange(start_val, end_val)
        splits.append((train_idx, val_idx))
        start_val = end_val
    return splits



def make_splits_wf(T: int, P_train: float = 0.5, num_windows: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walkforward validation (bloques sucesivos)."""
    L_train0 = int(P_train * T)
    if L_train0 <= 0 or L_train0 >= T:
        return []

    L_blk = (T - L_train0) // num_windows
    if L_blk <= 0:
        return []

    splits = []
    start = L_train0
    for _ in range(num_windows):
        end = start + L_blk
        if end > T:
            break
        tr_idx = np.arange(0, start)
        va_idx = np.arange(start, end)
        splits.append((tr_idx, va_idx))
        start = end
    return splits