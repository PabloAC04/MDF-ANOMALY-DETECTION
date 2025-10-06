import numpy as np
import pandas as pd
import cupy as cp
from cupyx.scipy.ndimage import median_filter as median_filter_gpu
from scipy.ndimage import median_filter as median_filter_cpu
import cudf


def generate_synthetic_timeseries(
    n: int = 1000,
    train_ratio: float = 0.4,
    val_ratio: float = 0.3,
    anomaly_ratio: float = 0.05,
    seed: int = 42
):
    """
    Genera tres conjuntos sintéticos (train, validation, test) con señales diversas.
    Las anomalías se inyectan automáticamente en validation y test con diferentes patrones.
    """
    np.random.seed(seed)
    t = np.linspace(0, 20, n)

    f1 = np.sin(0.5 * t) + 0.1 * np.random.randn(n)        # sinusoide
    f2 = np.sign(np.sin(3 * t)) + 0.2 * np.random.randn(n) # cuadrada
    f3 = np.cumsum(np.random.randn(n) * 0.05)              # random walk

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)

    df_train = pd.DataFrame({
        "timestamp": idx_train, "f1": f1[idx_train], "f2": f2[idx_train],
        "f3": f3[idx_train], "anomaly": 0
    })
    df_val = pd.DataFrame({
        "timestamp": idx_val, "f1": f1[idx_val], "f2": f2[idx_val],
        "f3": f3[idx_val], "anomaly": 0
    })
    df_test = pd.DataFrame({
        "timestamp": idx_test, "f1": f1[idx_test], "f2": f2[idx_test],
        "f3": f3[idx_test], "anomaly": 0
    })

    def inject_anomalies(df, anomaly_ratio):
        """Inyecta anomalías de tipo spike, drift, flat o noise en las señales."""
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
                if i + dur < df.index[-1]:
                    for col in affected:
                        slope = np.random.uniform(-3, 3)
                        drift = np.linspace(0, slope, dur)
                        df.loc[i:i + dur - 1, col] += drift
                    df.loc[i:i + dur - 1, "anomaly"] = 1
                    continue

            elif tipo == "flat":
                dur = np.random.randint(3, 15)
                if i + dur < df.index[-1]:
                    for col in affected:
                        if np.random.rand() < 0.5:
                            df.loc[i:i + dur - 1, col] = 0
                        else:
                            df.loc[i:i + dur - 1, col] = df.loc[i, col]
                    df.loc[i:i + dur - 1, "anomaly"] = 1
                    continue

            elif tipo == "noise":
                dur = np.random.randint(5, 20)
                if i + dur < df.index[-1]:
                    for col in affected:
                        noise = np.random.normal(0, 3, dur)
                        df.loc[i:i + dur - 1, col] += noise
                    df.loc[i:i + dur - 1, "anomaly"] = 1
                    continue

            df.loc[i, "anomaly"] = 1

        return df

    df_val = inject_anomalies(df_val, anomaly_ratio)
    df_test = inject_anomalies(df_test, anomaly_ratio)

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def hampel_cpu(
    x: np.ndarray,
    window: int = 25,
    sigma: float = 5.0,
    repair: str = "median",
    shrink_lambda: float = 0.7
):
    """
    Filtro de Hampel vectorizado en CPU.
    Reemplaza valores anómalos usando la mediana local o reducción proporcional.
    """
    k = 1.4826
    med = median_filter_cpu(x, size=window, mode="reflect")
    mad = k * median_filter_cpu(np.abs(x - med), size=window, mode="reflect")
    z = np.abs(x - med) / (mad + 1e-9)
    mask = z > sigma

    x_clean = x.copy()
    if repair == "median":
        x_clean[mask] = med[mask]
    elif repair == "shrink":
        x_clean[mask] = (1.0 - shrink_lambda) * x_clean[mask]

    return x_clean, mask


def hampel_gpu(
    x: cp.ndarray,
    window: int = 25,
    sigma: float = 5.0,
    repair: str = "median",
    shrink_lambda: float = 0.7
):
    """
    Filtro de Hampel vectorizado en GPU.
    Implementación equivalente a la versión CPU usando CuPy.
    """
    k = 1.4826
    med = median_filter_gpu(x, size=window, mode="reflect")
    mad = k * median_filter_gpu(cp.abs(x - med), size=window, mode="reflect")
    z = cp.abs(x - med) / (mad + 1e-9)
    mask = z > sigma

    x_clean = x.copy()
    if repair == "median":
        x_clean[mask] = med[mask]
    elif repair == "shrink":
        x_clean[mask] = (1.0 - shrink_lambda) * x_clean[mask]

    return x_clean, mask
