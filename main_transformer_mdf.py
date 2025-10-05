#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from modelos.TransformerAutoencoderDetector import TransformerAutoencoderDetector

# Soporte opcional para cuDF si está disponible
try:
    import cudf
    HAS_CUDF = True
except Exception:
    HAS_CUDF = False

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# ===== IMPORTA TU DETECTOR =====
# Si ya lo tienes en modelos/TransformerAutoencoderDetector.py:
# from modelos.TransformerAutoencoderDetector import TransformerAutoencoderDetector, create_windows
# En caso contrario, pego un "stub" mínimo de create_windows aquí para que sea autocontenido:
def create_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = X.shape
    if N < seq_len:
        raise ValueError(f"N ({N}) < seq_len ({seq_len}): no hay suficientes filas para crear ventanas")
    return np.stack([X[i:i + seq_len] for i in range(N - seq_len + 1)])


# ========= PUNTO CLAVE: CONTRIBUCIONES POR SEÑAL =========
def feature_contributions(x_last, recon, eps: float = 1e-12):
    """
    Calcula la contribución por señal a la reconstrucción.
    Acepta tanto tensores de PyTorch como arrays de NumPy.
    """
    # Asegurarse de que ambos estén en NumPy
    if isinstance(x_last, torch.Tensor):
        x_last = x_last.detach().cpu().numpy()
    if isinstance(recon, torch.Tensor):
        recon = recon.detach().cpu().numpy()

    # error cuadrático por-feature
    per_feat = (x_last - recon) ** 2  # [B, F]
    contrib_abs = per_feat
    denom = np.sum(contrib_abs, axis=1, keepdims=True) + eps
    contrib_norm = contrib_abs / denom
    return contrib_abs, contrib_norm



def is_datetime_series(s: pd.Series) -> bool:
    try:
        pd.to_datetime(s.iloc[:10], errors="raise")  # muestra
        return True
    except Exception:
        return False


def load_parquet(path: str):
    if HAS_CUDF:
        try:
            return cudf.read_parquet(path)
        except Exception:
            pass
    return pd.read_parquet(path)


def ensure_pandas(df):
    if HAS_CUDF and isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df


def pick_numeric_columns(df: pd.DataFrame, drop_cols=("timestamp", "split", "label")):
    drop_cols = [c for c in drop_cols if c in df.columns]
    num_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number, "float32", "float64", "int64", "int32"])
    feature_names = list(num_df.columns)
    return num_df, feature_names

def scores_and_recon(detector, X_seq):
    """
    Devuelve:
      - scores: errores de reconstrucción [N]
      - x_last: últimos timesteps reales [N, F]
      - recon: reconstrucciones [N, F]
    """
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(detector.device)
    detector.model.eval()
    with torch.no_grad():
        recon = detector.model(X_tensor)
        errors = torch.mean((X_tensor[:, -1, :] - recon) ** 2, dim=1).cpu().numpy()

    x_last = X_tensor[:, -1, :].cpu().numpy()
    recon = recon.cpu().numpy()
    return errors, x_last, recon


def main():
    parser = argparse.ArgumentParser(description="Experimento Transformer Autoencoder en .parquet de MDF")
    parser.add_argument("--parquet", type=str, required=True, help="Ruta al data.parquet")
    parser.add_argument("--seq_len", type=int, default=10, help="Tamaño de ventana")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train_frac", type=float, default=0.6, help="Fracción de entrenamiento (resto test)")
    parser.add_argument("--outdir", type=str, default="outputs_transformer_mdf", help="Carpeta de salida")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--plot_name", type=str, default="scores_plot.png")
    parser.add_argument("--json_name", type=str, default="anomalies_with_contributions.json")
    parser.add_argument("--scores_csv", type=str, default="scores.csv")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ==== CARGA ====
    df_all = load_parquet(args.parquet)
    df_all = ensure_pandas(df_all)

    # Timestamp: usa 'timestamp' si existe y es convertible; si no, index secuencial
    if "timestamp" in df_all.columns and is_datetime_series(df_all["timestamp"]):
        ts = pd.to_datetime(df_all["timestamp"])
    else:
        ts = pd.Series(pd.to_datetime(np.arange(len(df_all)), unit="s"))  # <---- Serie

    # Selecciona split si existe (pero aquí se pide entrenar/test del mismo parquet, 60/40 temporal)
    # Por simplicidad ignoramos 'split' y hacemos corte temporal directo.
    num_df, feature_names = pick_numeric_columns(df_all)

    if len(feature_names) == 0:
        raise ValueError("No se encontraron columnas numéricas para entrenar.")

    X_all = num_df.to_numpy(dtype=np.float32)

    N = len(X_all)
    N_train = int(N * args.train_frac)
    if N_train <= args.seq_len:
        raise ValueError(f"train_frac produce muy pocas filas: N_train={N_train}, seq_len={args.seq_len}")

    X_train_raw = X_all[:N_train]
    X_test_raw = X_all[N_train:]
    ts_train = ts[:N_train].reset_index(drop=True)
    ts_test = ts[N_train:].reset_index(drop=True)

    # Instancia detector
    det = TransformerAutoencoderDetector(
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        seq_len=args.seq_len, device=None, verbose=True, use_scaler=True
    )

    # ====== ENTRENAMIENTO ======
    X_train_seq = det.preprocess(X_train_raw, retrain=True)
    det.fit(X_train_seq)

    # ====== TEST: SCORES + RECON PARA CONTRIBUCIONES ======
    X_test_seq = det.preprocess(X_test_raw, retrain=False)
    scores, x_last, recon = scores_and_recon(det, X_test_seq)  # scores longitud = len(X_test_raw) - seq_len + 1

    # Alinear timestamps de test con ventanas (cada score corresponde al final de la ventana)
    ts_test_eff = ts_test.iloc[args.seq_len - 1:].reset_index(drop=True)
    assert len(ts_test_eff) == len(scores), "Desalineación entre timestamps y scores"

    # Contribuciones por-feature (absolutas y normalizadas)
    contrib_abs, contrib_norm = feature_contributions(x_last, recon)

    # Predicciones binarias
    preds = (scores > det.threshold).astype(int)

    # ====== SALIDAS ======
    # 1) CSV con scores
    scores_df = pd.DataFrame({
        "timestamp": ts_test_eff.astype(str),
        "score": scores,
        "is_anomaly": preds
    })
    scores_csv_path = os.path.join(args.outdir, args.scores_csv)
    scores_df.to_csv(scores_csv_path, index=False)

    # 2) JSON con anomalías y contribuciones
    anomalies = []
    for i in np.where(preds == 1)[0]:
        ts_i = ts_test_eff.iloc[i]
        # top contribuciones ordenadas desc
        order = np.argsort(-contrib_norm[i])
        sigs = [
            {"name": feature_names[j], "contribution": float(contrib_norm[i, j])}
            for j in order if contrib_norm[i, j] > 0.0
        ]
        anomalies.append({
            "timestamp": pd.to_datetime(ts_i).isoformat(),
            "score": float(scores[i]),
            "signals": sigs
        })

    # Ordenar las anomalías de mayor a menor score
    anomalies = sorted(anomalies, key=lambda x: x["score"], reverse=True)

    json_obj = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "parquet": os.path.abspath(args.parquet),
            "seq_len": args.seq_len,
            "train_frac": args.train_frac,
            "n_features": len(feature_names),
            "threshold": float(det.threshold)
        },
        "threshold": float(det.threshold),
        "anomalies": anomalies
    }

    json_path = os.path.join(args.outdir, args.json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)

    # 3) Plot de scores con umbral y anomalías
    plt.figure(figsize=(12, 5))
    plt.plot(ts_test_eff, scores, label="score")
    plt.axhline(det.threshold, linestyle="--", label="umbral")
    if len(anomalies) > 0:
        anom_idx = np.where(preds == 1)[0]
        plt.scatter(ts_test_eff.iloc[anom_idx], scores[anom_idx], marker="o", s=25, label="anomalía")
    plt.title("Scores de reconstrucción (test)")
    plt.xlabel("timestamp")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, args.plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # === 4) Graficar las 10 anomalías con mayor score (señales completas) ===
    top_k = 10
    top_anomalies = anomalies[:top_k]
    os.makedirs(os.path.join(args.outdir, "top_anomalies"), exist_ok=True)

    # Convertir reconstrucciones y reales a DataFrame para acceder por índice
    df_real = pd.DataFrame(x_last, columns=feature_names)
    df_recon = pd.DataFrame(recon, columns=feature_names)

    for idx, anom in enumerate(top_anomalies, start=1):
        ts_i = pd.to_datetime(anom["timestamp"])
        score_i = anom["score"]

        # Buscar el índice correspondiente en los timestamps efectivos
        i = ts_test_eff[ts_test_eff == ts_i].index
        if len(i) == 0:
            continue
        i = i[0]

        # Tomar las 3 señales con mayor contribución a esta anomalía
        top_signals = [s["name"] for s in sorted(anom["signals"], key=lambda x: x["contribution"], reverse=True)[:3]]
        n_signals = len(top_signals)

        # Crear la figura
        plt.figure(figsize=(12, 2.5 * n_signals))
        for j, sig in enumerate(top_signals):
            plt.subplot(n_signals, 1, j + 1)
            plt.plot(ts_test_eff, df_real[sig], label=f"{sig} (real)", color='blue', linewidth=1)
            plt.plot(ts_test_eff, df_recon[sig], label=f"{sig} (reconstrucción)", color='orange', linewidth=1, alpha=0.8)
            plt.axvline(ts_i, color='red', linestyle='--', label='anomalía' if j == 0 else None)
            plt.ylabel(sig)
            if j == 0:
                plt.title(f"Anomalía #{idx} | Score={score_i:.5f}\n{ts_i}")
            if j == n_signals - 1:
                plt.xlabel("timestamp")
            plt.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        path_img = os.path.join(args.outdir, "top_anomalies", f"anomaly_{idx:02d}.png")
        plt.savefig(path_img, dpi=150)
        plt.close()



    print(f"[✓] Guardado plot: {plot_path}")
    print(f"[✓] Guardado JSON: {json_path}")
    print(f"[✓] Guardado CSV scores: {scores_csv_path}")
    print(f"[i] Umbral SPOT: {det.threshold:.6f} | nº anomalías: {int(preds.sum())}")


if __name__ == "__main__":
    main()
