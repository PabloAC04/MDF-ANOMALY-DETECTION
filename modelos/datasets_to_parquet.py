import pandas as pd
import os
import pyarrow
import fastparquet
import numpy as np
import ast

class DatasetsToParquet:
    
    def __init__(self, dataset_name: str, input_path = "D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION", output_path = "D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data"):
        self.dataset_name = dataset_name
        self.input_path = input_path
        self.output_path = output_path

        self.dataset_methods = {
            "BATADAL": self._process_batadal,
            "SKAB": self._process_skab,
            "WADI": self._process_wadi,
            "EbayRanSynCoders": self._process_EbayRanSynCoders,
            "SMAP": self._process_smap,
            "MSL": self._process_msl
        }
    
    def convert(self):
        self._load_and_process()

    def _load_and_process(self):

        func = self.dataset_methods.get(self.dataset_name)
        if func:
            return func()
        else:
            print(f"[!] Dataset '{self.dataset_name}' no reconocido.")
            return None

    def _process_batadal(self):
        print("[...] Procesando dataset BATADAL")

        dataset1 = os.path.join(self.input_path, "BATADAL", "BATADAL_dataset03.csv")  # train (sin anomalías)
        dataset2 = os.path.join(self.input_path, "BATADAL", "BATADAL_dataset04.csv")  # con anomalías

        # Cargar
        df1 = pd.read_csv(dataset1, sep=",", decimal=".")
        df2 = pd.read_csv(dataset2, sep=",", decimal=".")

        # Quitar espacios de cabeceras
        df1.columns = df1.columns.str.strip()
        df2.columns = df2.columns.str.strip()

        # Renombrar columnas
        if "DATETIME" in df1.columns:
            df1.rename(columns={"DATETIME": "timestamp"}, inplace=True)
        if "DATETIME" in df2.columns:
            df2.rename(columns={"DATETIME": "timestamp"}, inplace=True)
        if "ATT_FLAG" in df1.columns:
            df1.drop(columns=["ATT_FLAG"], inplace=True)  # no tiene sentido en train
        if "ATT_FLAG" in df2.columns:
            df2.rename(columns={"ATT_FLAG": "anomaly"}, inplace=True)

        # Cambiar -999 -> 0 SOLO en anomaly
        if "anomaly" in df2.columns:
            df2["anomaly"] = df2["anomaly"].replace(-999, 0)

        # --- Añadir columna 'split' ---
        df1["anomaly"] = 0  # dataset1 es training sin anomalías
        df1["split"] = "train"

        validation_size = int(len(df2) * 0.6)
        validation_df = df2.iloc[:validation_size].copy()
        test_df = df2.iloc[validation_size:].copy()

        validation_df["split"] = "val"
        test_df["split"] = "test"

        # Concatenar todo
        df_all = pd.concat([df1, validation_df, test_df], axis=0).reset_index(drop=True)

        # Guardar en único parquet
        folder_save = os.path.join(self.output_path, "BATADAL")
        os.makedirs(folder_save, exist_ok=True)
        self._save_to_parquet(df_all, os.path.join(folder_save, "data.parquet"))

        print(f"[✓] Dataset BATADAL procesado y guardado en {folder_save}/data.parquet")

    def _process_skab(self):
        print("[...] Procesando dataset SKAB")

        dataset_train = os.path.join(self.input_path, "SKAB", "anomaly-free", "anomaly-free.csv")

        # Cargar dataset de entrenamiento
        df_train = pd.read_csv(dataset_train, sep=";", decimal=".")
        df_train.rename(columns={"DATETIME": "timestamp"}, inplace=True)

        df_train["anomaly"] = 0   # entrenamiento sin anomalías
        df_train["split"] = "train"

        # Cargar datasets de test/val (concatenados)
        test_val_paths = [
            "0.csv", "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv",
            "8.csv", "9.csv", "10.csv", "11.csv", "12.csv", "13.csv", "14.csv", "15.csv"
        ]

        df_test_val = pd.DataFrame()

        for path in test_val_paths:
            full_path = os.path.join(self.input_path, "SKAB", "valve1", path)
            df_test_val_sub = pd.read_csv(full_path, sep=";", decimal=".")
            df_test_val_sub.rename(columns={"DATETIME": "timestamp"}, inplace=True)
            df_test_val = pd.concat([df_test_val, df_test_val_sub], ignore_index=True)

        # Quitar columna "changepoint" si existe
        if "changepoint" in df_test_val.columns:
            df_test_val.drop(columns=["changepoint"], inplace=True)

        # Split 60% val / 40% test
        validation_size = int(len(df_test_val) * 0.6)
        validation_df = df_test_val.iloc[:validation_size].copy()
        test_df = df_test_val.iloc[validation_size:].copy()

        validation_df["split"] = "val"
        test_df["split"] = "test"

        # Concatenar todo
        df_all = pd.concat([df_train, validation_df, test_df], axis=0).reset_index(drop=True)

        # Guardar único data.parquet
        folder_save = os.path.join(self.output_path, "SKAB")
        os.makedirs(folder_save, exist_ok=True)
        self._save_to_parquet(df_all, os.path.join(folder_save, "data.parquet"))

        print(f"[✓] Dataset SKAB procesado y guardado en {folder_save}/data.parquet")


    def _save_to_parquet(self, df, path):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        df.to_parquet(path, index=False)
        print(f"[✓] Guardado en: {path}")

    def _process_wadi(self):
        print("[...] Procesando dataset WADI")

        training_file = os.path.join(self.input_path, "WADI", "WADI_14days_new.csv")
        attack_file = os.path.join(self.input_path, "WADI", "WADI_attackdataLABLE.csv")

        # Leer datasets
        df_train = pd.read_csv(training_file, low_memory=False)
        df_attack = pd.read_csv(attack_file, skiprows=1, low_memory=False)

        # Limpiar nombres de columnas
        df_train.columns = df_train.columns.str.strip()
        df_attack.columns = df_attack.columns.str.strip()

        # Eliminar columnas innecesarias
        for df in [df_train, df_attack]:
            for col in ["Row", "Row "]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        # Crear columna 'timestamp'
        df_train["timestamp"] = pd.to_datetime(
            df_train["Date"] + " " + df_train["Time"],
            format="%m/%d/%y %M:%S.%f",
            errors="coerce"
        )
        df_attack["timestamp"] = pd.to_datetime(
            df_attack["Date"] + " " + df_attack["Time"],
            format="%m/%d/%y %M:%S.%f",
            errors="coerce"
        )

        # Eliminar columnas originales Date/Time
        df_train.drop(columns=["Date", "Time"], inplace=True)
        df_attack.drop(columns=["Date", "Time"], inplace=True)

        # Renombrar y ajustar etiquetas de anomalía
        label_col = "Attack LABLE (1:No Attack, -1:Attack)"
        if label_col in df_attack.columns:
            df_attack.rename(columns={label_col: "anomaly"}, inplace=True)
            df_attack["anomaly"] = df_attack["anomaly"].replace({1: 0, -1: 1})

        # Asegurar columna anomaly en df_train (entrenamiento sin ataques)
        df_train["anomaly"] = 0

        # Añadir columna 'split'
        df_train["split"] = "train"
        val_size = int(len(df_attack) * 0.6)
        df_val = df_attack.iloc[:val_size].copy()
        df_test = df_attack.iloc[val_size:].copy()
        df_val["split"] = "val"
        df_test["split"] = "test"

        # Concatenar todo en un solo DataFrame
        df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

        # Guardar único data.parquet
        folder_save = os.path.join(self.output_path, "WADI")
        os.makedirs(folder_save, exist_ok=True)
        self._save_to_parquet(df_all, os.path.join(folder_save, "data.parquet"))

        print(f"[✓] Dataset WADI procesado y guardado en {folder_save}/data.parquet")


    def _process_EbayRanSynCoders(self):
        print("[...] Procesando dataset EbayRanSynCoders")

        base_path = os.path.join(self.input_path, "RANSynCoders-main", "data")
        train_path = os.path.join(base_path, "train.csv")
        test_path = os.path.join(base_path, "test.csv")
        labels_path = os.path.join(base_path, "test_label.csv")

        # Leer archivos
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_labels = pd.read_csv(labels_path)

        # Renombrar columna de tiempo
        df_train.rename(columns={"timestamp_(min)": "timestamp"}, inplace=True)
        df_test.rename(columns={"timestamp_(min)": "timestamp"}, inplace=True)

        # Unir etiquetas de test
        df_test = df_test.merge(
            df_labels.rename(columns={"timestamp_(min)": "timestamp", "label": "anomaly"}),
            on="timestamp",
            how="left"
        )

        # Asegurar columna anomaly en train (sin anomalías conocidas)
        df_train["anomaly"] = 0

        # Añadir columna split
        df_train["split"] = "train"
        val_size = int(len(df_test) * 0.6)
        df_val = df_test.iloc[:val_size].copy()
        df_test_final = df_test.iloc[val_size:].copy()
        df_val["split"] = "val"
        df_test_final["split"] = "test"

        # Concatenar todo en un único DataFrame
        df_all = pd.concat([df_train, df_val, df_test_final], axis=0).reset_index(drop=True)

        # Guardar en un único data.parquet
        folder_save = os.path.join(self.output_path, "EbayRanSynCoders")
        os.makedirs(folder_save, exist_ok=True)
        self._save_to_parquet(df_all, os.path.join(folder_save, "data.parquet"))

        print(f"[✓] Dataset EbayRanSynCoders procesado y guardado en {folder_save}/data.parquet")


    def _process_smap(self):
        print("[...] Procesando dataset SMAP (carpeta compartida)")
        return self._process_smap_msl_generic(
            dataset_folder="SMAP & MSL",
            spacecraft="SMAP",
            exclude_channels={"P-2"}  # como en muchos repos
        )

    def _process_msl(self):
        print("[...] Procesando dataset MSL (carpeta compartida)")
        return self._process_smap_msl_generic(
            dataset_folder="SMAP & MSL",
            spacecraft="MSL",
            exclude_channels=set()
        )

    def _process_smap_msl_generic(self, dataset_folder: str, spacecraft: str, exclude_channels: set):
        """
        Entrada (compartida):
            {input_path}/{dataset_folder}/
                ├── labeled_anomalies.csv  (spacecraft ∈ {SMAP, MSL})
                └── data/data/
                    ├── train/*.npy
                    └── test/*.npy

        Salida (un único data.parquet con splits):
            {output_path}/{spacecraft}/data.parquet
        """
        root = os.path.join(self.input_path, dataset_folder)
        data_dir = os.path.join(root, "data", "data")
        train_dir = os.path.join(data_dir, "train")
        test_dir  = os.path.join(data_dir, "test")

        lab_csv = os.path.join(root, "labeled_anomalies.csv")
        if not os.path.exists(lab_csv):
            raise FileNotFoundError(f"No se encuentra {lab_csv}")

        df_lab_all = pd.read_csv(lab_csv)
        df_lab = (
            df_lab_all[df_lab_all["spacecraft"].astype(str).str.upper() == spacecraft.upper()]
            .copy()
            .sort_values("chan_id")
        )

        def _parse_ranges(s):
            seq = ast.literal_eval(str(s).replace("'", '"'))
            out = []
            for pair in seq:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a, b = int(pair[0]), int(pair[1])
                    if b < a: a, b = b, a
                    out.append((a, b))
            return out

        ranges_dict, len_dict = {}, {}
        for _, row in df_lab.iterrows():
            chan = str(row["chan_id"])
            ranges_dict.setdefault(chan, []).extend(_parse_ranges(row["anomaly_sequences"]))
            if "num_values" in df_lab.columns:
                len_dict[chan] = int(row["num_values"])

        valid_chans_csv_order = [c for c in df_lab["chan_id"].astype(str).tolist() if c not in exclude_channels]

        train_present = {os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.lower().endswith(".npy")}
        test_present  = {os.path.splitext(f)[0] for f in os.listdir(test_dir)  if f.lower().endswith(".npy")}
        train_chans = [c for c in valid_chans_csv_order if c in train_present]
        test_chans  = [c for c in valid_chans_csv_order if c in test_present]

        def _load_npy_float32(path):
            arr = np.load(path)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr.astype(np.float32, copy=False)

        # TRAIN
        train_parts = []
        for chan in train_chans:
            arr = _load_npy_float32(os.path.join(train_dir, chan + ".npy"))
            df = pd.DataFrame(arr, columns=[f"f{i+1}" for i in range(arr.shape[1])])
            df.insert(0, "timestamp", np.arange(len(df), dtype=np.int64))
            df.insert(1, "channel", chan)
            df["anomaly"] = 0
            df["spacecraft"] = spacecraft
            df["split"] = "train"
            train_parts.append(df)
        df_train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()

        # TEST
        test_parts = []
        for chan in test_chans:
            arr = _load_npy_float32(os.path.join(test_dir, chan + ".npy"))
            T = arr.shape[0]
            lab = np.zeros(T, dtype=bool)
            for (ini, fin) in ranges_dict.get(chan, []):
                ini = max(0, ini); fin = min(T - 1, fin)
                if fin >= ini:
                    lab[ini:fin+1] = True
            df = pd.DataFrame(arr, columns=[f"f{i+1}" for i in range(arr.shape[1])])
            df.insert(0, "timestamp", np.arange(T, dtype=np.int64))
            df.insert(1, "channel", chan)
            df["anomaly"] = lab.astype(np.int8)
            df["spacecraft"] = spacecraft
            test_parts.append(df)

        df_test_full = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
        cut = int(len(df_test_full) * 0.6)
        df_val  = df_test_full.iloc[:cut].copy()
        df_test = df_test_full.iloc[cut:].copy()
        df_val["split"] = "val"
        df_test["split"] = "test"

        # Concatenar todo
        df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

        # Guardar único data.parquet
        outdir = os.path.join(self.output_path, spacecraft)
        os.makedirs(outdir, exist_ok=True)
        self._save_to_parquet(df_all, os.path.join(outdir, "data.parquet"))

        print(f"[✓] Dataset {spacecraft} procesado y guardado en {outdir}/data.parquet")



def inspect_parquet(folder: str, split: str = None, n: int = 5):
    """
    Lee un único data.parquet en `folder` y muestra información básica.
    
    Args:
        folder (str): Ruta a la carpeta donde está el parquet (ej: data/BATADAL).
        split (str): Uno de {"train", "val", "test"} o None para todo el dataset.
        n (int): Número de filas a mostrar como ejemplo.
    
    Returns:
        df (pd.DataFrame): El DataFrame cargado (filtrado por split si aplica).
    """
    path = os.path.join(folder, "data.parquet")
    if not os.path.exists(path):
        print(f"[!] No existe el archivo {path}")
        return None

    df = pd.read_parquet(path)

    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()

    print(f"\n[✓] {('todo el dataset' if split is None else split.upper())} cargado desde {path}")
    print(f"   - Filas: {len(df)}")
    print(f"   - Columnas: {list(df.columns)}")
    if "anomaly" in df.columns:
        print(f"   - Conteo anomalías: {df['anomaly'].sum()} de {len(df)} "
              f"({100*df['anomaly'].mean():.2f}%)")
    print("\n[Vista previa]")
    print(df.head(n))
    return df


def load_project_parquets(folder: str, splits: bool = False):
    """
    Carga un único data.parquet. 
    
    Args:
        folder (str): Ruta a la carpeta donde está el parquet (ej: data/BATADAL).
        splits (bool): 
            - False (por defecto): devuelve el DataFrame completo.
            - True: devuelve (df_train, df_val, df_test) según la columna 'split'.
    
    Returns:
        - df (pd.DataFrame) si splits=False
        - (df_train, df_val, df_test) si splits=True
    """
    path = os.path.join(folder, "data.parquet")
    if not os.path.exists(path):
        print(f"[!] No existe el archivo {path}")
        return None if not splits else (None, None, None)

    df = pd.read_parquet(path)
    print(f"[✓] DATA cargado desde {path} ({len(df)} filas)")

    if not splits:
        return df

    if "split" not in df.columns:
        raise ValueError("El parquet no contiene columna 'split' necesaria para separar.")

    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test"].copy()

    print(f"   - Train: {len(df_train)} filas")
    print(f"   - Val:   {len(df_val)} filas")
    print(f"   - Test:  {len(df_test)} filas")

    return df_train, df_val, df_test


if __name__ == "__main__":

    dataset = "BATADAL"

    DatasetsToParquet(
        dataset_name=dataset,
    ).convert()

    inspect_parquet(f'D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data\SMAP')
    inspect_parquet(f'D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data\MSL')
    inspect_parquet(f'D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data\EbayRanSynCoders')
    inspect_parquet(f'D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data\SKAB')
    inspect_parquet(f'D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data\BATADAL')
    inspect_parquet(f'D:\TFG\TFG\Avance\MDF-ANOMALY-DETECTION\modelos\data\WADI')

    # data = load_project_parquets("data/WADI")

    # #Visualizar datos

    # for split, df in data.items():
    #     if df is not None:
    #         print(f"\n--- {split.upper()} ---")
    #         print(df.head())
    #         print(f"Filas: {len(df)}, Columnas: {list(df.columns)}")






