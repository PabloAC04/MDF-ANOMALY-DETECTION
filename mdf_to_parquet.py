import os
import json
import pandas as pd
import numpy as np
from asammdf import MDF
import concurrent.futures
import multiprocessing

def get_sampling_period(sig):
    ts = sig.timestamps
    if len(ts) > 1:
        diffs = np.diff(ts)
        return float(np.mean(diffs))   # en segundos
    return None

def canales_validos(mdf_file):
    """
    Selecciona canales válidos:
    - Quita 't' y 'time'
    - Prefiere ICanal frente a Canal
    - Evita duplicados tras el último '.'
    """
    canales_k = list(mdf_file.channels_db.keys())
    canales_finales = {}

    for canal in canales_k:
        if canal in ("t", "time"):
            continue

        nombre_corto = canal.split('.')[-1]
        base = nombre_corto[1:] if nombre_corto.startswith("I") else nombre_corto

        if base not in canales_finales:
            canales_finales[base] = canal
        else:
            if nombre_corto.startswith("I") and not canales_finales[base].split('.')[-1].startswith("I"):
                canales_finales[base] = canal

    return list(canales_finales.values())

def remove_constant_columns(df: pd.DataFrame):
    """Elimina columnas donde todos los valores son iguales (incluyendo si todos son NaN)."""
    nunique = df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index
    if len(constant_cols) > 0:
        print(f"[INFO] Eliminadas {len(constant_cols)} columnas constantes")
    return df.drop(columns=constant_cols)

def process_channel_chunk(path_mdf, channels, raster, include_labels=False):
    """Procesa un subconjunto de canales usando mdf.filter y devuelve un DataFrame y, opcionalmente, mappings categóricos."""
    mdf = MDF(path_mdf)
    mdf_reduced = mdf.filter(channels)

    df_raw = mdf_reduced.to_dataframe(channels=channels, raster=raster, raw=True)
    cat_map = {}

    if include_labels:
        df_lbl = mdf_reduced.to_dataframe(channels=channels, raster=raster, raw=False)
        for col in df_raw.columns:
            if col in df_lbl.columns:
                # si todos los valores de label son numéricos → ignorar
                lbl_vals = df_lbl[col].dropna().unique()

                # ignorar si todos los valores de label son numéricos
                if all([pd.api.types.is_number(x) for x in lbl_vals]):
                    continue
                
                # ignorar si solo hay 1 valor distinto de NaN
                if len(lbl_vals) <= 1:
                    continue

                mapping = {}
                for r, l in zip(df_raw[col], df_lbl[col]):
                    if pd.isna(r) or pd.isna(l):
                        continue
                    mapping[str(r)] = str(l)

                if mapping:
                    cat_map[col] = mapping

    # eliminar NaN del dataframe antes de devolver
    df_raw = df_raw.ffill().bfill().fillna(0)
    df_raw = remove_constant_columns(df_raw)

    return df_raw, cat_map


def export_parquet(df, path_out, periods):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df.to_parquet(path_out, engine="pyarrow")

    # Guardar periodo mínimo
    valid = [p for p in periods if p is not None]
    min_period = min(valid) if valid else None
    with open(path_out.replace(".parquet", "_meta.json"), "w") as f:
        json.dump({"min_period": min_period}, f)

    print(f"[FIN] Guardado en {path_out}, shape={df.shape}, periodo={min_period}")

def mdf_to_parquet_by_channels(path_mdf, out_dir, num_workers=None):
    """
    Convierte un MDF/MF4 en varios Parquet:
    - general
    - base
    - uno por conjunto
    Además guarda un JSON con los canales categóricos (raw→label).
    """
    mdf = MDF(path_mdf)
    all_channels = canales_validos(mdf)

    # Detectar periodos
    periods = []
    for ch in all_channels:
        try:
            sig = mdf.get(ch)
            p = get_sampling_period(sig)
            if p is not None:
                periods.append(p)
        except Exception:
            pass
    global_min_period = min(periods) if periods else None
    print(f"[INFO] Periodo mínimo global: {global_min_period}")
    print(f"[INFO] Canales seleccionados: {len(all_channels)}")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    print(f"[INFO] Usando {num_workers} workers")

    # Dividir canales en lotes
    chunk_size = int(np.ceil(len(all_channels) / num_workers))
    channel_chunks = [all_channels[i:i + chunk_size] for i in range(0, len(all_channels), chunk_size)]

    # ---- General ----
    dfs = []
    categoricos_total = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_channel_chunk, path_mdf, chunk, global_min_period, True)
            for chunk in channel_chunks
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            df_chunk, cat_map = f.result()
            dfs.append(df_chunk)
            categoricos_total.update(cat_map)
            print(f"[OK] Chunk {i+1}/{len(channel_chunks)} procesado ({df_chunk.shape[1]} columnas)")

    df_general = pd.concat(dfs, axis=1)
    export_parquet(df_general, os.path.join(out_dir, "general.parquet"), periods)

    # Guardar categóricos en JSON
    if categoricos_total:
        with open(os.path.join(out_dir, "general_categoricals.json"), "w") as f:
            json.dump(categoricos_total, f, indent=2)
        print(f"[INFO] Guardado mapping de categóricos en general_categoricals.json")
    else:
        print("[INFO] No se detectaron canales categóricos")

    # ---- Base ----
    base_channels = []
    conjuntos = {}
    for ch in all_channels:
        sig = mdf.get(ch)
        if hasattr(sig.samples, "dtype") and sig.samples.dtype.names:
            root = ".".join(ch.split('.')[:-1])
            conjuntos.setdefault(root, []).append(ch)
        else:
            base_channels.append(ch)

    base_periods = [get_sampling_period(mdf.get(ch)) for ch in base_channels]
    base_min_period = min([p for p in base_periods if p is not None], default=global_min_period)

    df_base = mdf.to_dataframe(channels=base_channels, raster=base_min_period, raw=True)
    df_base.ffill().bfill().fillna(0)
    df_base = remove_constant_columns(df_base)
    export_parquet(df_base, os.path.join(out_dir, "base.parquet"), base_periods)

    # ---- Conjuntos ----
    for root, chans in conjuntos.items():
        set_periods = [get_sampling_period(mdf.get(ch)) for ch in chans]
        set_min_period = min([p for p in set_periods if p is not None], default=global_min_period)

        df_set = mdf.to_dataframe(channels=chans, raster=set_min_period, raw=True)
        df_set.ffill().bfill().fillna(0)
        df_set = remove_constant_columns(df_set)
        fname = os.path.basename(root).replace(".", "_") + ".parquet"
        export_parquet(df_set, os.path.join(out_dir, fname), set_periods)

def info_parquet(path_parquet: str, max_cols: int = 10):
    # Leer el parquet
    df = pd.read_parquet(path_parquet).astype('float32')

    # Número de filas y columnas
    filas, columnas = df.shape

    # Tamaño en memoria (bytes → MB)
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024**2)

    print(f"📂 Archivo: {path_parquet}")
    print(f"📊 Filas: {filas:,}")
    print(f"📊 Columnas: {columnas:,}")
    print(f"💾 Tamaño en memoria: {mem_mb:.2f} MB")
    print(f"💾 Tamaño en disco (aprox.): {os.path.getsize(path_parquet) / (1024**2):.2f} MB")

    # Tipos de datos
    print("\n📑 Tipos de datos:")
    print(df.dtypes.value_counts())

    # Estadísticas resumidas
    print("\n📈 Resumen estadístico de columnas numéricas:")
    stats = df.describe().T[["mean", "std", "min", "max"]]

    # Mostrar solo algunas columnas si son demasiadas
    if len(stats) > max_cols:
        print(stats.head(max_cols))
        print(f"... ({len(stats) - max_cols} columnas más)")
    else:
        print(stats)

    return df



if __name__ == "__main__":
    # archivo_mdf = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDF1.mf4"
    # salida_parquet = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF1"

    # df = mdf_to_parquet_by_channels(archivo_mdf, salida_parquet, num_workers=8)

    # archivo_mdf = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDF2.mf4"
    # salida_parquet = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF2"

    # df = mdf_to_parquet_by_channels(archivo_mdf, salida_parquet, num_workers=8)

    # archivo_mdf = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDF3.mf4"
    # salida_parquet = "/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF3"

    # df = mdf_to_parquet_by_channels(archivo_mdf, salida_parquet, num_workers=8)

    df = info_parquet("/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF1/general.parquet")
    df = info_parquet("/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF1/base.parquet")
    df = info_parquet("/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF1/CAN_CH-FD.parquet")
    df = info_parquet("/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF1/CAN_ITS2-FD.parquet")
    df = info_parquet("/home/pablo/TFG/MDF-ANOMALY-DETECTION/MDFs/MDF1/CAN_ITS3-FD.parquet")