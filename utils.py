import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from statsmodels.tsa.seasonal import STL

def generate_synthetic_timeseries(
    n: int = 1000,
    train_ratio: float = 0.4,
    val_ratio: float = 0.3,
    anomaly_ratio: float = 0.05,
    seed: int = 42
):
    """
    Genera tres DataFrames sintéticos (train, validation, test) con señales más diversas.
    Las anomalías se inyectan automáticamente en validation y test, con formas más variadas.
    """
    np.random.seed(seed)
    t = np.linspace(0, 20, n)

    # Señales distintas
    f1 = np.sin(0.5*t) + 0.1*np.random.randn(n)               # sinusoide
    f2 = np.sign(np.sin(3*t)) + 0.2*np.random.randn(n)        # cuadrada
    f3 = np.cumsum(np.random.randn(n) * 0.05)                 # random walk

    # Cortes
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    idx_train = np.arange(0, n_train)
    idx_val   = np.arange(n_train, n_train + n_val)
    idx_test  = np.arange(n_train + n_val, n)

    # Crear DataFrames
    df_train = pd.DataFrame({"timestamp": idx_train, "f1": f1[idx_train], "f2": f2[idx_train], "f3": f3[idx_train], "anomaly": 0})
    df_val   = pd.DataFrame({"timestamp": idx_val,   "f1": f1[idx_val],   "f2": f2[idx_val],   "f3": f3[idx_val],   "anomaly": 0})
    df_test  = pd.DataFrame({"timestamp": idx_test,  "f1": f1[idx_test],  "f2": f2[idx_test],  "f3": f3[idx_test],  "anomaly": 0})

    def inject_anomalies(df, anomaly_ratio):
        n_anom = max(1, int(len(df) * anomaly_ratio))
        idx_anom = np.random.choice(df.index, size=n_anom, replace=False)

        for i in idx_anom:
            tipo = np.random.choice(["spike", "drift", "flat", "noise"])
            affected = np.random.choice(["f1", "f2", "f3"],
                                        size=np.random.randint(1, 4),
                                        replace=False)

            if tipo == "spike":
                for col in affected:
                    amp = np.random.uniform(2, 8) * np.random.choice([-1, 1])
                    df.loc[i, col] += amp

            elif tipo == "drift":
                dur = np.random.randint(5, 30)
                if i+dur < df.index[-1]:
                    for col in affected:
                        slope = np.random.uniform(-3, 3)
                        drift = np.linspace(0, slope, dur)
                        df.loc[i:i+dur-1, col] += drift
                    df.loc[i:i+dur-1, "anomaly"] = 1
                    continue

            elif tipo == "flat":
                dur = np.random.randint(3, 15)
                if i+dur < df.index[-1]:
                    for col in affected:
                        if np.random.rand() < 0.5:
                            df.loc[i:i+dur-1, col] = 0
                        else:
                            df.loc[i:i+dur-1, col] = df.loc[i, col]
                    df.loc[i:i+dur-1, "anomaly"] = 1
                    continue

            elif tipo == "noise":
                dur = np.random.randint(5, 20)
                if i+dur < df.index[-1]:
                    for col in affected:
                        noise = np.random.normal(0, 3, dur)
                        df.loc[i:i+dur-1, col] += noise
                    df.loc[i:i+dur-1, "anomaly"] = 1
                    continue

            df.loc[i, "anomaly"] = 1

        return df

    df_val  = inject_anomalies(df_val, anomaly_ratio)
    df_test = inject_anomalies(df_test, anomaly_ratio)

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def hampel_on_residual(
    data: Union[pd.Series, pd.DataFrame],
    seasonal_period: Optional[int] = None,
    stl_robust: bool = True,
    window: int = 25,
    sigma: float = 5.0,
    causal: bool = False,
    max_repair_ratio: float = 0.02,
    repair: str = "median",          # 'median' | 'interp' | 'shrink'
    shrink_lambda: float = 0.7,      # usado si repair='shrink'
    impute_for_stl: str = "interpolate",  # 'interpolate' | 'ffill' | 'none'
    return_mask: bool = False
) -> Union[pd.Series, pd.DataFrame, Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]]:
    """
    Limpia outliers aplicando *filtro de Hampel* sobre el RESIDUO de una señal.
    - Si seasonal_period se especifica, se usa STL para separar tendencia+estacionalidad y
      se aplica Hampel al residuo (recomendado para señales estacionales).
    - Si no, se aplica Hampel directamente a la serie (equivale a residuo=serie).
    
    Parámetros
    ----------
    data : Series o DataFrame
        Señal (o señales) a limpiar.
    seasonal_period : int o None
        Periodo estacional para STL (p. ej., 24, 48, 7*24, etc.). Si None, no se usa STL.
    stl_robust : bool
        STL robusta frente a outliers al estimar T+S.
    window : int
        Tamaño de ventana para mediana/MAD (impar recomendado). Si causal=True, ventana “pasada”.
    sigma : float
        Umbral de Hampel en unidades de MAD (típico 5.0).
    causal : bool
        Si True, rolling no centrado (solo pasado). Si False, centrado (mejor “suavizado” offline).
    max_repair_ratio : float
        Máximo % de puntos a reparar por columna y bloque (ej. 0.02 = 2%). Se reparan solo los peores.
        Si 0 -> sin límite.
    repair : str
        Estrategia de reparación de puntos marcados:
        - 'median': reemplaza por la mediana local (suave y robusto, recomendado).
        - 'interp': interpola linealmente el residuo entre puntos no anómalos.
        - 'shrink': mezcla hacia 0 el residuo: r <- (1-lambda)*r   (lambda en [0,1]).
    shrink_lambda : float
        Factor de “encogimiento” si repair='shrink' (0.7 por defecto).
    impute_for_stl : str
        Manejo de NaNs antes de STL: 'interpolate', 'ffill' o 'none'. STL no acepta NaNs.
    return_mask : bool
        Si True, devuelve también la máscara booleana de outliers (mismo shape que data).
    
    Returns
    -------
    cleaned : Series o DataFrame (mismo tipo que 'data')
        Señal limpia.
    mask : Series o DataFrame (opcional)
        Máscara de outliers detectados (True = outlier).
    """
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    df = df.astype(float)

    center = not causal

    def _stl_decompose(y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        if seasonal_period is None:
            # Sin STL: todo es "residuo"
            return np.zeros_like(y.values), np.zeros_like(y.values)
        y_stl = y.copy()
        if y_stl.isna().any():
            if impute_for_stl == "interpolate":
                y_stl = y_stl.interpolate(limit_direction="both")
            elif impute_for_stl == "ffill":
                y_stl = y_stl.ffill().bfill()
            # si 'none', STL puede fallar si hay NaN
        res = STL(y_stl.values, period=seasonal_period, robust=stl_robust).fit()
        return res.trend, res.seasonal

    def _rolling_median(series: pd.Series, center: bool) -> pd.Series:
        return series.rolling(window=window, min_periods=1, center=center).median()

    def _rolling_mad(series: pd.Series, med: pd.Series, center: bool) -> pd.Series:
        abs_dev = (series - med).abs()
        return abs_dev.rolling(window=window, min_periods=1, center=center).median()

    cleaned_cols = {}
    mask_cols = {}

    for col in df.columns:
        y = df[col]

        # 1) Descomposición si corresponde
        trend, seas = _stl_decompose(y)
        resid = y.values - trend - seas

        # 2) Hampel sobre residuo
        resid_s = pd.Series(resid, index=y.index)
        med = _rolling_median(resid_s, center=center)
        mad = _rolling_mad(resid_s, med, center=center)
        scale = 1.4826 * mad.replace(0, np.nan)
        z = (resid_s - med).abs() / scale
        mask = (z > sigma).fillna(False)


        # 3) Si hay límite de reparación, quedarnos con los peores 'top_n'
        if max_repair_ratio and max_repair_ratio > 0:
            n = len(resid_s)
            top_n = int(np.ceil(max_repair_ratio * n))
            if mask.sum() > top_n:
                # ordenar por severidad y mantener solo los top_n
                sev = (z.fillna(0)).values
                idx_sorted = np.argsort(-sev)  # desc
                idx_top = set(idx_sorted[:top_n])
                mask = pd.Series([i in idx_top and m for i, m in enumerate(mask.values)],
                                 index=mask.index)

        # 4) Reparación del residuo
        resid_clean = resid_s.copy()
        if mask.any():
            if repair == "median":
                resid_clean[mask] = med[mask]
            elif repair == "interp":
                # Interpola SOLO en puntos marcados (sobre residuo)
                idx = np.arange(len(resid_clean))
                good = ~mask
                if good.sum() >= 2:
                    interp_vals = np.interp(idx[mask], idx[good], resid_clean[good])
                    resid_clean[mask] = interp_vals
                else:
                    resid_clean[mask] = med[mask]
            elif repair == "shrink":
                resid_clean[mask] = (1.0 - shrink_lambda) * resid_clean[mask]
            else:
                raise ValueError("repair debe ser 'median' | 'interp' | 'shrink'.")

        # 5) Reconstrucción
        y_clean = trend + seas + resid_clean.values
        cleaned_cols[col] = pd.Series(y_clean, index=y.index)
        mask_cols[col] = mask

    cleaned = pd.concat(cleaned_cols, axis=1)
    cleaned.columns = df.columns
    if is_series:
        cleaned = cleaned.iloc[:, 0]
    mask_df = pd.concat(mask_cols, axis=1)
    mask_df.columns = df.columns
    if is_series:
        mask_df = mask_df.iloc[:, 0]

    return (cleaned, mask_df) if return_mask else cleaned
