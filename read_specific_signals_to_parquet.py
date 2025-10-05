from asammdf import MDF
import pandas as pd
import numpy as np
import os
import json

def read_specific_signals_to_parquet(
    path_mdf: str,
    out_path: str,
    signals_wanted: list,
    raster: float = None
):
    """
    Lee señales específicas desde un MDF/MF4, interpola y exporta a parquet + JSON.
    """

    # Abrir archivo MDF
    mdf = MDF(path_mdf)
    print(f"[INFO] Archivo cargado: {os.path.basename(path_mdf)}")

    # Buscar canales que coincidan con los nombres deseados
    all_channels = list(mdf.channels_db.keys())
    selected = [ch for ch in all_channels if any(ch.endswith(sig) for sig in signals_wanted)]

    if not selected:
        print("[WARN] No se encontraron canales que coincidan con los nombres deseados.")
        return

    print(f"[INFO] {len(selected)} canales seleccionados:")
    for ch in selected:
        print(f"   - {ch}")

    # Determinar periodo de muestreo (promedio)
    def get_sampling_period(sig):
        ts = sig.timestamps
        return np.mean(np.diff(ts)) if len(ts) > 1 else None

    if raster is None:
        periods = []
        for ch in selected:
            try:
                sig = mdf.get(ch)
                p = get_sampling_period(sig)
                if p is not None:
                    periods.append(p)
            except Exception:
                pass
        raster = min(periods) if periods else None
        print(f"[INFO] Periodo mínimo estimado: {raster:.6f} s")

    # Convertir a DataFrame con interpolación temporal
    df = mdf.to_dataframe(channels=selected, raster=raster, time_as_date=True, raw=True)
    df = df.interpolate(method='time').ffill().bfill()

    # Eliminar columnas constantes
    nunique = df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index
    if len(const_cols) > 0:
        print(f"[INFO] Eliminadas {len(const_cols)} columnas constantes: {list(const_cols)}")
        df = df.drop(columns=const_cols)

    # Guardar como parquet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow")
    print(f"[OK] Guardado parquet en {out_path} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    # Guardar metadatos JSON
    meta = {
        "signals": df.columns.tolist(),
        "raster": raster,
        "source_file": os.path.basename(path_mdf)
    }
    with open(out_path.replace(".parquet", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Guardado metadatos en {out_path.replace('.parquet', '_meta.json')}")

    return df


# === Uso ===
if __name__ == "__main__":
    archivo_mdf = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDF1.mf4"
    salida = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MotorSignals/Motor.parquet"

    señales_bsw = [
        "EngineRPM",
        "EngineCoolantTemp",
        "EngineWaterPumpStatus"
    ]

    df = read_specific_signals_to_parquet(archivo_mdf, salida, señales_bsw)
