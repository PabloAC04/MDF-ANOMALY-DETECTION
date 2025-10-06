import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


def _binary_to_intervals(y):
    """Convierte una secuencia binaria (0/1) en intervalos [inicio, fin] inclusivos."""
    y = np.asarray(y).astype(int)
    if y.size == 0:
        return []
    dif = np.diff(np.pad(y, (1, 1), constant_values=0))
    starts = np.where(dif == 1)[0]
    ends = np.where(dif == -1)[0] - 1
    return list(zip(starts, ends))


def _plot_intervals(ax, intervals, y, height=1.0, color='tab:red', alpha=0.3, label=None):
    """Dibuja regiones sombreadas para representar intervalos de anomalías o detecciones."""
    for (a, b) in intervals:
        ax.axvspan(a, b, color=color, alpha=alpha, ymin=0, ymax=height, label=label)
        label = None  # Evita duplicar la etiqueta en la leyenda


def plot_pr_curve(y_true, y_score, title="Precision–Recall"):
    """Dibuja la curva Precision–Recall y calcula el área aproximada bajo la curva."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = np.trapz(prec[::-1], rec[::-1])
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f"AP≈{ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.show()


def plot_roc_curve(y_true, y_score, title="ROC"):
    """Dibuja la curva ROC y calcula el área bajo la curva (AUC)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.show()


def plot_confmat(y_true, y_pred, title="Matriz de confusión"):
    """Muestra la matriz de confusión con anotaciones de conteo."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['Real 0', 'Real 1'])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_score_hist(y_true, y_score, threshold=None, title="Distribución de scores"):
    """Muestra la distribución de scores de normales vs anómalos."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    plt.figure(figsize=(6, 5))
    plt.hist(y_score[y_true == 0], bins=50, alpha=0.6, label="Normal")
    plt.hist(y_score[y_true == 1], bins=50, alpha=0.6, label="Anómalo")
    if threshold is not None:
        plt.axvline(threshold, ls="--", label=f"Umbral={threshold:.3f}")
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("conteo")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.show()


def plot_timeline_coverage(y_true, y_pred, y_score=None, title="Cobertura temporal"):
    """Visualiza la coincidencia temporal entre anomalías reales y detectadas."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    T = len(y_true)

    true_int = _binary_to_intervals(y_true)
    pred_int = _binary_to_intervals(y_pred)

    fig, ax = plt.subplots(figsize=(12, 3))
    _plot_intervals(ax, true_int, y_true, color='tab:red', alpha=0.25, label="Anomalías reales")
    _plot_intervals(ax, pred_int, y_pred, color='tab:blue', alpha=0.25, label="Detecciones")

    if y_score is not None:
        ax2 = ax.twinx()
        ax2.plot(np.arange(T), y_score, lw=1.2, alpha=0.8, label="score")
        ax2.set_ylabel("score")

    ax.set_xlim(0, T - 1)
    ax.set_xlabel("t")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(True, axis="x", ls="--", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_window_coverage_bars(y_true, y_pred, title="Cobertura por evento"):
    """Muestra, para cada evento real, si existe intersección con alguna detección."""
    true_int = _binary_to_intervals(y_true)
    pred_int = _binary_to_intervals(y_pred)

    def covered(interval, preds):
        a, b = interval
        for (p, q) in preds:
            if not (q < a or p > b):
                return 1
        return 0

    cov = [covered(iv, pred_int) for iv in true_int]
    if len(cov) == 0:
        print("No hay eventos reales -> cobertura vacía.")
        return

    plt.figure(figsize=(min(12, 0.4 * len(cov) + 2), 3.5))
    plt.bar(range(len(cov)), cov)
    plt.yticks([0, 1], ["No", "Sí"])
    plt.xlabel("Evento real")
    plt.title(title + f"  (tasa={np.mean(cov):.2f})")
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
