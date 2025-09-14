import pandas as pd
import os
import pyarrow
import fastparquet
import numpy as np
import ast

class DatasetsToParquet:
    
    def __init__(self, dataset_name: str, input_path: str, output_path: str):
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
            df1.drop(columns=["ATT_FLAG"], inplace=True)
        if "ATT_FLAG" in df2.columns:
            df2.rename(columns={"ATT_FLAG": "anomaly"}, inplace=True)

        # Cambiar -999 -> 0 SOLO en anomaly
        if "anomaly" in df2.columns:
            df2["anomaly"] = df2["anomaly"].replace(-999, 0)

        # Guardar TRAIN
        folder_save = os.path.join(self.output_path, "BATADAL")
        os.makedirs(folder_save, exist_ok=True)
        self._save_to_parquet(df1, os.path.join(folder_save, "training.parquet"))

        # Dividir en validación (60%) y test (40%)
        validation_size = int(len(df2) * 0.6)
        validation_df = df2.iloc[:validation_size]
        test_df = df2.iloc[validation_size:]

        # Guardar VAL y TEST
        self._save_to_parquet(validation_df, os.path.join(folder_save, "validation.parquet"))
        self._save_to_parquet(test_df, os.path.join(folder_save, "test.parquet"))


    def _process_skab(self):
        print("[...] Procesando dataset SKAB")

        dataset_train = os.path.join(self.input_path, "SKAB", "anomaly-free", "anomaly-free.csv")

        # Cambiar el nombre de la columna DATETIME a timestamp
        df_train = pd.read_csv(dataset_train, sep=";", decimal=".")

        df_train.rename(columns={"DATETIME": "timestamp"}, inplace=True)

        # Guardar el dataset de entrenamiento
        folder_save = os.path.join(self.output_path, "SKAB")
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
        training_path = os.path.join(folder_save, "training.parquet")
        self._save_to_parquet(df_train, training_path)

        test_val_paths = ["0.csv", "1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv", "11.csv", "12.csv", "13.csv", "14.csv", "15.csv"]

        df_test_val = pd.DataFrame()

        for path in test_val_paths:
            full_path = os.path.join(self.input_path, "SKAB", "valve1", path)
            df_test_val_sub = pd.read_csv(full_path, sep=";", decimal=".")

            # Cambiar el nombre de la columna DATETIME a timestamp
            df_test_val_sub.rename(columns={"DATETIME": "timestamp"}, inplace=True)
            

            df_test_val = pd.concat([df_test_val, df_test_val_sub], ignore_index=True)

        if "changepoint" in df_test_val.columns:
            df_test_val.drop(columns=["changepoint"], inplace=True)
        # Guardar el dataset de validación y test 60% 40%
        
        validation_size = int(len(df_test_val) * 0.6)
        validation_df = df_test_val.iloc[:validation_size]
        test_df = df_test_val.iloc[validation_size:]

        # Guardar el dataset de validación
        validation_path = os.path.join(folder_save, "validation.parquet")
        self._save_to_parquet(validation_df, validation_path)
        # Guardar el dataset de test
        test_path = os.path.join(folder_save, "test.parquet")
        self._save_to_parquet(test_df, test_path)

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

        # Limpiar nombres de columnas para evitar errores con espacios
        df_train.columns = df_train.columns.str.strip()
        df_attack.columns = df_attack.columns.str.strip()


        # Eliminar columna "Row" o "Row " si está presente
        for df in [df_train, df_attack]:
            for col in ["Row", "Row "]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        # Crear columna 'timestamp' uniendo 'Date' y 'Time'
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

        # Renombrar la columna de etiquetas de anomalía
        label_col = "Attack LABLE (1:No Attack, -1:Attack)"
        if label_col in df_attack.columns:
            df_attack.rename(columns={label_col: "anomaly"}, inplace=True)
            df_attack["anomaly"] = df_attack["anomaly"].replace({1: 0, -1: 1})  # <-- AJUSTE CLAVE AQUÍ

        # Crear carpeta de salida
        folder_save = os.path.join(self.output_path, "WADI")
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        # Guardar dataset de entrenamiento
        training_path = os.path.join(folder_save, "training.parquet")
        self._save_to_parquet(df_train, training_path)

        # Dividir dataset de ataques en validación y test
        val_size = int(len(df_attack) * 0.6)
        df_val = df_attack.iloc[:val_size]
        df_test = df_attack.iloc[val_size:]

        # Guardar validación
        validation_path = os.path.join(folder_save, "validation.parquet")
        self._save_to_parquet(df_val, validation_path)

        # Guardar test
        test_path = os.path.join(folder_save, "test.parquet")
        self._save_to_parquet(df_test, test_path)

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
        df_test = df_test.merge(df_labels.rename(columns={"timestamp_(min)": "timestamp", "label": "anomaly"}), on="timestamp", how="left")

        # Crear carpeta de salida
        folder_save = os.path.join(self.output_path, "EbayRanSynCoders")
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        # Guardar entrenamiento
        training_path = os.path.join(folder_save, "training.parquet")
        self._save_to_parquet(df_train, training_path)

        # Dividir test en validación (60%) y test (40%)
        val_size = int(len(df_test) * 0.6)
        df_val = df_test.iloc[:val_size]
        df_test_final = df_test.iloc[val_size:]

        # Guardar validación
        validation_path = os.path.join(folder_save, "validation.parquet")
        self._save_to_parquet(df_val, validation_path)

        # Guardar test
        test_path = os.path.join(folder_save, "test.parquet")
        self._save_to_parquet(df_test_final, test_path)

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
                    ├── train/*.npy        (mezclados)
                    └── test/*.npy         (mezclados)

            Salida (separada por nave):
                {output_path}/{spacecraft}/training.parquet
                {output_path}/{spacecraft}/validation.parquet
                {output_path}/{spacecraft}/test.parquet
        """
        root = os.path.join(self.input_path, dataset_folder)
        data_dir = os.path.join(root, "data", "data")
        train_dir = os.path.join(data_dir, "train")
        test_dir  = os.path.join(data_dir, "test")

        lab_csv = os.path.join(root, "labeled_anomalies.csv")
        if not os.path.exists(lab_csv):
            raise FileNotFoundError(f"No se encuentra {lab_csv}")

        df_lab_all = pd.read_csv(lab_csv)
        # Filtra por nave y ordena por chan_id como hace el script de referencia
        df_lab = (
            df_lab_all[df_lab_all["spacecraft"].astype(str).str.upper() == spacecraft.upper()]
            .copy()
            .sort_values("chan_id")
        )

        # Agrega rangos por canal (por si hay filas repetidas de un mismo chan)
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

        # Canales válidos por CSV (y exclusiones)
        valid_chans_csv_order = [c for c in df_lab["chan_id"].astype(str).tolist() if c not in exclude_channels]
        valid_set = set(valid_chans_csv_order)

        # Archivos presentes y canales disponibles (intersección con válidos, preserva orden del CSV)
        train_present = {os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.lower().endswith(".npy")}
        test_present  = {os.path.splitext(f)[0] for f in os.listdir(test_dir)  if f.lower().endswith(".npy")}
        train_chans = [c for c in valid_chans_csv_order if c in train_present]
        test_chans  = [c for c in valid_chans_csv_order if c in test_present]

        def _load_npy_float32(path):
            arr = np.load(path)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr.astype(np.float32, copy=False)

        # TRAIN concatenado (anomaly=0)
        train_parts = []
        for chan in train_chans:
            arr = _load_npy_float32(os.path.join(train_dir, chan + ".npy"))
            if chan in len_dict and len_dict[chan] != arr.shape[0]:
                print(f"[!] Longitud distinta en {chan} (csv={len_dict[chan]}, npy={arr.shape[0]})")
            df = pd.DataFrame(arr, columns=[f"f{i+1}" for i in range(arr.shape[1])])
            df.insert(0, "timestamp", np.arange(len(df), dtype=np.int64))
            df.insert(1, "channel", chan)
            df["anomaly"] = 0
            df["spacecraft"] = spacecraft
            train_parts.append(df)
        df_train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()

        # TEST concatenado con etiquetas (rangos inclusivos) y split 60/40 por orden global (CSV)
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

        # Guardado por nave
        outdir = os.path.join(self.output_path, spacecraft)
        os.makedirs(outdir, exist_ok=True)
        self._save_to_parquet(df_train, os.path.join(outdir, "training.parquet"))
        self._save_to_parquet(df_val,   os.path.join(outdir, "validation.parquet"))
        self._save_to_parquet(df_test,  os.path.join(outdir, "test.parquet"))


def inspect_parquet(folder: str, split: str = "training", n: int = 5):
    """
    Lee un parquet guardado en `folder` y muestra información básica.

    Args:
        folder (str): Ruta a la carpeta donde están los parquet (ej: data/SMAP).
        split (str): Uno de {"training", "validation", "test"}.
        n (int): Número de filas a mostrar como ejemplo.

    Returns:
        df (pd.DataFrame): El dataframe cargado.
    """
    path = os.path.join(folder, f"{split}.parquet")
    if not os.path.exists(path):
        print(f"[!] No existe el archivo {path}")
        return None

    df = pd.read_parquet(path)

    print(f"\n[✓] {split.upper()} cargado desde {path}")
    print(f"   - Filas: {len(df)}")
    print(f"   - Columnas: {list(df.columns)}")
    if "anomaly" in df.columns:
        print(f"   - Conteo anomalías: {df['anomaly'].sum()} de {len(df)} "
            f"({100*df['anomaly'].mean():.2f}%)")
    print("\n[Vista previa]")
    print(df.head(n))
    return df

def load_project_parquets(folder: str):
    """
    Carga los tres splits (training, validation, test) de un dataset en formato parquet.

    Args:
        folder (str): Ruta a la carpeta donde están los parquet (ej: data/SMAP).

    Returns:
        dict: Diccionario con DataFrames para cada split.
    """
    splits = ["training", "validation", "test"]
    data = {}
    for split in splits:
        path = os.path.join(folder, f"{split}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            data[split] = df
            print(f"[✓] {split} cargado desde {path} ({len(df)} filas)")
        else:
            print(f"[!] No existe el archivo {path}")
            data[split] = None
    return data["training"], data["validation"], data["test"]

if __name__ == "__main__":

    dataset = "WADI"

    DatasetsToParquet(
        dataset_name=dataset,
        input_path="../",  
        output_path="./data"
    ).convert()

    inspect_parquet(f'data/{dataset}', "training")
    inspect_parquet(f'data/{dataset}', "validation")
    inspect_parquet(f'data/{dataset}', "test")

    # data = load_project_parquets("data/WADI")

    # #Visualizar datos

    # for split, df in data.items():
    #     if df is not None:
    #         print(f"\n--- {split.upper()} ---")
    #         print(df.head())
    #         print(f"Filas: {len(df)}, Columnas: {list(df.columns)}")






