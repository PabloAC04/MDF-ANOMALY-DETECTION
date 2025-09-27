from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def precision_metric(y_true, y_pred, y_score=None):
    return precision_score(y_true, y_pred, zero_division=0)

def recall_metric(y_true, y_pred, y_score=None):
    return recall_score(y_true, y_pred, zero_division=0)

def f1_metric(y_true, y_pred, y_score=None):
    return f1_score(y_true, y_pred, zero_division=0)

def roc_auc_metric(y_true, y_pred, y_score):
    # Nota: requiere scores continuos, no solo etiquetas
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan

def scaled_sigmoid(x):
    """Función scaledSigmoid de NAB."""
    if x > 3.0:  # FP muy lejos del final de la ventana
        return -1.0
    return 2.0 / (1.0 + np.exp(5.0 * x)) - 1.0


def nab_reward(t_det, start, end, A_TP=1.0):
    """
    Recompensa NAB para una detección en t_det dentro de [start, end].
    """
    window_width = (end - start + 1)
    # posición relativa: -1 en inicio de ventana, 0 en final
    pos = -(end - t_det + 1) / float(window_width)
    return A_TP * scaled_sigmoid(pos)


def extend_window(start, end, n, ratio=0.2, min_ext=1, max_ext=None):
    """Extiende la ventana de anomalía al estilo NAB."""
    length = end - start + 1
    delta = max(min_ext, int(ratio * length))
    if max_ext is None:
        max_ext = int(0.1 * n)
    delta = min(delta, max_ext)

    s_ext = max(0, start - delta)
    e_ext = min(n - 1, end + delta)
    return s_ext, e_ext


def nab_metric(y_true, y_pred, y_score=None,
               A_TP=1.0, A_FP=0.11, A_FN=1.0,
               ratio=0.2, min_ext=5, max_ext=None,
               normalize=True, clip=True):
    """
    Implementación del NAB score con ventanas extendidas y función oficial.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    assert len(y_pred) == n, "y_true e y_pred deben tener la misma longitud"

    score_raw = 0.0

    # --- localizar ventanas de anomalía ---
    anomalies = []
    in_anomaly = False
    for i in range(n):
        if y_true[i] == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif y_true[i] == 0 and in_anomaly:
            anomalies.append((start, i - 1))
            in_anomaly = False
    if in_anomaly:
        anomalies.append((start, n - 1))

    # --- extender ventanas ---
    extended_anomalies = [extend_window(s, e, n, ratio, min_ext, max_ext)
                          for (s, e) in anomalies]

    # máscara de "dentro de ventana" para contar FP
    inside_mask = np.zeros(n, dtype=bool)
    for (s, e) in extended_anomalies:
        inside_mask[s:e + 1] = True

    # --- calcular recompensa ---
    for (start, end) in extended_anomalies:
        detections = np.where((y_pred[start:end + 1] == 1))[0]
        if detections.size > 0:
            t_det = start + detections[0]
            reward = nab_reward(t_det, start, end, A_TP=A_TP)
            score_raw += reward
        else:
            score_raw -= A_FN

    # --- falsos positivos ---
    num_fp = int(np.sum((y_pred == 1) & (~inside_mask)))
    score_raw -= A_FP * num_fp

    if not normalize:
        return score_raw

    # --- Normalización estilo NAB ---
    W = len(extended_anomalies)
    if W > 0:
        S_best = A_TP * W
        S_null = -A_FN * W
        denom = S_best - S_null
        score_norm = (score_raw - S_null) / denom if denom > 0 else np.nan
        if clip and np.isfinite(score_norm):
            score_norm = max(-1.0, min(1.0, score_norm))
        return score_norm
    else:
        fp_rate = num_fp / max(1, n)
        score_norm = 1.0 - fp_rate
        if clip:
            score_norm = max(-1.0, min(1.0, score_norm))
        return score_norm



def window_coverage_metric(y_true, y_pred, y_score=None):
    """
        Mide % de cobertura por ventana de anomalía.
    """
    n = len(y_true)
    coverages = []

    # localizar ventanas
    anomalies = []
    in_anomaly = False
    for i in range(n):
        if y_true[i] == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif y_true[i] == 0 and in_anomaly:
            anomalies.append((start, i-1))
            in_anomaly = False
    if in_anomaly:
        anomalies.append((start, n-1))

    for (start, end) in anomalies:
        window_len = end - start + 1
        hits = sum(y_pred[start:end+1])
        coverage = hits / window_len if window_len > 0 else 0.0
        coverages.append(coverage)

    return np.mean(coverages) if coverages else 0.0
