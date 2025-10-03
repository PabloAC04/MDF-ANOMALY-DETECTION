import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
from IPython.display import display
from tqdm import tqdm
from modelos.datasets_to_parquet import load_project_parquets
import os
import numpy as np
import cudf
import cupy as cp

from modelos.PCAAnomalyDetector import candidate_n_components
from sklearn.decomposition import PCA as PCAsk

from modelos.ValidationPipeline import ValidationPipelineGPU, ValidationPipelineCPU
from visualization import (
    plot_pr_curve, plot_roc_curve, plot_confmat,
    plot_score_hist, plot_timeline_coverage, plot_window_coverage_bars
)

from dask import delayed, compute
from dask.diagnostics import ProgressBar


@delayed
def run_one_config(model_class, kwargs, X_trainval, y_trainval, device, metrics, params_cv, mode, hampel_cfg):
    model = model_class(**kwargs)
    pipeline = get_pipeline(
        model=model,
        metrics=metrics,
        mode=mode,
        params_cv=params_cv,
        hampel_cfg=hampel_cfg,
        device=device,
        X=X_trainval
    )
    results = pipeline.validate(X_trainval, y_trainval)
    results.update(kwargs)
    return results

def get_pipeline(model, metrics, mode, params_cv, hampel_cfg, device, X):
    """
    Devuelve el pipeline CPU/GPU en función del parámetro device.
    - device="cpu": fuerza CPU
    - device="gpu": fuerza GPU
    """
    if device == "gpu":
        return ValidationPipelineGPU(model, metrics, mode, params_cv, hampel_cfg)
    else:
        return ValidationPipelineCPU(model, metrics, mode, params_cv, hampel_cfg)

def run_experiment(
    model_class,
    param_grid,
    X_trainval, y_trainval,
    X_test, y_test,
    metrics,
    params_cv,
    device: str = "cpu",
    mode="tscv",
    hampel_cfg={"window": 25, "sigma": 5.0},
    top_k=5,
    sort_metric="nab",
    plot_mode="best"
):
    grid_results = []

    # 1) Grid search en validación
    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))

    tasks = [
        run_one_config(model_class, dict(zip(keys, combo)),
                    X_trainval, y_trainval, device,
                    metrics, params_cv, mode, hampel_cfg)
        for combo in combos
    ]

    with ProgressBar():
        grid_results = compute(*tasks)  # ejecución en paralelo

    df_results = pd.DataFrame(grid_results)
    metric_cols = list(metrics.keys())
    param_cols = list(param_grid.keys())
    column_order = metric_cols + param_cols
    df_results = df_results[column_order]

    # 2) Top-k por métrica de validación
    df_sorted = df_results.sort_values(by=sort_metric, ascending=False)
    topk = df_sorted.head(top_k)
    topk = topk[column_order]

    # 3) Evaluación en test para el top-k
    # (preprocesado fuera del bucle para consistencia)
    model_tmp = model_class(**{k: topk.iloc[0][k] for k in param_cols})
    X_trainval_proc = model_tmp.preprocess(X_trainval, retrain=True)
    X_test_proc     = model_tmp.preprocess(X_test, retrain=False)

    final_rows = []
    best_plots_payload = None  # guardaremos y_true/y_pred/y_score de la mejor config

    for i, (_, row) in enumerate(tqdm(topk.iterrows(), total=len(topk), desc="Evaluación en test")):
        kwargs = {k: row[k] for k in param_cols}
        if mode == "walkforward" and "epochs" in kwargs and "num_windows" in params_cv:
            kwargs["epochs"] = kwargs["epochs"] * params_cv["num_windows"]

        model = model_class(**kwargs)
        model.fit(X_trainval_proc)

        y_pred  = model.predict(X_test_proc, y_test)
        y_score = model.anomaly_score(X_test_proc, y_test)

        if isinstance(y_pred, tuple):
            y_pred, y_test = y_pred
        if isinstance(y_score, tuple):
            y_score, y_test = y_score

        res = kwargs.copy()

        if device == "gpu":
            y_test_cpu = cp.asnumpy(cp.asarray(y_test))
            y_pred_cpu = cp.asnumpy(y_pred)
            y_score_cpu = cp.asnumpy(y_score)
        else:
            y_test_cpu = np.asarray(y_test)
            y_pred_cpu = np.asarray(y_pred)
            y_score_cpu = np.asarray(y_score)

        for name, metric in metrics.items():
            res[name] = metric(y_test_cpu, y_pred_cpu, y_score_cpu)
        final_rows.append(res)

        # guardamos payload de la mejor config (orden en topk ya es el de validación)
        if i == 0:
            best_plots_payload = (y_test_cpu, y_pred_cpu, y_score_cpu, kwargs)

        # plot on the fly si se pide "all"
        if plot_mode == "all":
            _plot_all(y_test_cpu, y_pred_cpu, y_score_cpu, model_class.__name__, kwargs)

    df_final = pd.DataFrame(final_rows)[column_order]

    # 4) Mostrar tablas
    pd.set_option("display.max_columns", None)

    print("="*60)
    print(f"Top {top_k} configuraciones (ordenadas por {sort_metric} en validación):")
    print("="*60)
    display(topk.round(3))

    print("="*60)
    print("Resultados finales en TEST (top-k configs):")
    print("="*60)
    display(df_final.round(3))

    # 5) Plot solo de la mejor configuración (limpio y suficiente)
    if plot_mode == "best" and best_plots_payload is not None:
        y_true, y_pred, y_score, best_kwargs = best_plots_payload
        _plot_all(y_true, y_pred, y_score, model_class.__name__, best_kwargs)

    return df_results, topk, df_final


def _plot_all(y_true, y_pred, y_score, model_name, kwargs):
    title_suffix = f"{model_name} {kwargs}"

    # 1) PR curve
    plot_pr_curve(y_true, y_score, title=f"PR – {title_suffix}")

    # 2) ROC curve
    plot_roc_curve(y_true, y_score, title=f"ROC – {title_suffix}")

    # 3) Confusion matrix
    plot_confmat(y_true, y_pred, title=f"Matriz de confusión – {title_suffix}")

    # 4) Score hist (si tienes umbral accesible, pásalo aquí; si no, omite)
    #    Para IF puedes estimar el umbral como quantile (1 - contamination) si no hay atributo en el modelo.
    threshold = None
    if "contamination" in kwargs and kwargs["contamination"] is not None:
        q = 1.0 - float(kwargs["contamination"])
        threshold = np.quantile(y_score, q)
    plot_score_hist(y_true, y_score, threshold=threshold, title=f"Distribución de scores – {title_suffix}")

    # 5) Timeline de cobertura + barras por evento
    plot_timeline_coverage(y_true, y_pred, y_score, title=f"Cobertura temporal – {title_suffix}")
    plot_window_coverage_bars(y_true, y_pred, title=f"Cobertura por evento – {title_suffix}")

DATA_DIR = r"/home/pablo/TFG/MDF-ANOMALY-DETECTION/modelos/data"

def decide_device(X):
    if cp.cuda.runtime.getDeviceCount() > 0 and len(X) > 1.5e5:
        return "gpu"
    else:
        return "cpu"

def run_dataset_experiment(
    dataset_name: str,
    model_class,
    param_grid,
    metrics,
    params_cv,
    device: str = "cpu",
    mode: str = "tscv",
    hampel_cfg: dict = {"window": 25, "sigma": 5.0},
    top_k: int = 5,
    sort_metric: str = "nab",
    plot_mode: str = "best"
):
    """
    Ejecuta un experimento de anomalía sobre un dataset completo.

    Args
    ----
    dataset_name : str
        Nombre de la carpeta del dataset (ej: "BATADAL", "SMAP").
    model_class : class
        Clase del detector (ej: IsolationForest).
    param_grid : dict
        Grid de hiperparámetros.
    metrics : dict
        Diccionario de métricas {nombre: función}.
    params_cv : dict
        Parámetros de validación cruzada temporal.
    data_dir : str
        Ruta raíz donde están los parquets (ej: "D:/.../modelos/data").
    mode, seasonal_period, hampel_cfg, top_k, sort_metric, plot_mode
        Igual que en run_experiment.
    """

    print("="*80)
    print(f"🏁 Ejecutando experimento en dataset: {dataset_name}")
    print("="*80)

    folder = os.path.join(DATA_DIR, dataset_name)
    df_train, df_val, df_test = load_project_parquets(folder, splits=True)

    if device != "cpu" and device != "gpu":
        device = decide_device(df_train)

    if device == "gpu":
        print("Usando GPU (cuDF + CuPy) para este experimento.")
        df_train = cudf.DataFrame.from_pandas(df_train)
        df_val   = cudf.DataFrame.from_pandas(df_val)
        df_test  = cudf.DataFrame.from_pandas(df_test)
    else:
        print("Usando CPU (pandas + numpy) para este experimento.")

    # Features = todas menos columnas reservadas
    feature_cols = [c for c in df_train.columns if c not in ["split", "timestamp", "anomaly"]]

    # Conjuntos
    if device == "gpu":
        df_trainval = cudf.concat([df_train, df_val], ignore_index=True)
    else:
        df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    
    X_trainval, y_trainval = df_trainval[feature_cols], df_trainval["anomaly"]
    X_test, y_test = df_test[feature_cols], df_test["anomaly"]

    #   Calcular n_components candidatos si se pide
    if "n_components" in param_grid:
        X_train = df_train[feature_cols]
        X_train_cpu = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
        
        print("→ Calculando n_components candidatos automáticamente con candidate_n_components...")
        pca_tmp = PCAsk(svd_solver="full")
        pca_tmp.fit(X_train_cpu)
        param_grid["n_components"] = candidate_n_components(pca_tmp)
        print(f"Candidatos generados: {param_grid['n_components']}")


    # Calcular P_train real según splits
    P_train = len(df_train) / (len(df_train) + len(df_val))
    params_cv = {**params_cv, "P_train": P_train}

    # Ejecutar el experimento
    df_results, topk, df_final = run_experiment(
        model_class=model_class,
        param_grid=param_grid,
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        X_test=X_test,
        y_test=y_test,
        metrics=metrics,
        params_cv=params_cv,
        device=device,
        mode=mode,
        hampel_cfg=hampel_cfg,
        top_k=top_k,
        sort_metric=sort_metric,
        plot_mode=plot_mode
    )


    return df_results, topk, df_final