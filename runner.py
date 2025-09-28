import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
from IPython.display import display

from modelos.ValidationPipeline import ValidationPipeline

def run_experiment(
    model_class,
    param_grid,
    X_trainval, y_trainval,
    X_test, y_test,
    metrics,
    params_cv,
    mode="tscv",
    seasonal_period="auto",
    hampel_cfg={"window": 25, "sigma": 5.0},
    top_k=5,
    sort_metric="nab",
    plot_results=True,
    df_test=None   # para los gráficos de señales
):
    grid_results = []

    # 1. Grid search
    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        kwargs = dict(zip(keys, combo))
        model = model_class(**kwargs)

        pipeline = ValidationPipeline(
            model=model,
            metrics=metrics,
            mode=mode,
            params=params_cv,
            seasonal_period=seasonal_period,
            hampel_cfg=hampel_cfg
        )

        results = pipeline.validate(X_trainval, y_trainval)
        results.update(kwargs)
        grid_results.append(results)

    df_results = pd.DataFrame(grid_results)

    # 2. Selección top-k
    df_sorted = df_results.sort_values(by=sort_metric, ascending=False)
    topk = df_sorted.head(top_k)

    # 3. Evaluación en test
    final_results = []
    for _, row in topk.iterrows():
        kwargs = {k: row[k] for k in param_grid.keys()}
        model = model_class(**kwargs)
        model.fit(X_trainval)

        y_pred = model.predict(X_test)
        y_score = model.anomaly_score(X_test)

        res = kwargs.copy()
        for name, metric in metrics.items():
            res[name] = metric(y_test.astype(int), y_pred, y_score)
        final_results.append(res)

        if plot_results:
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc_val = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {model_class.__name__} {kwargs}")
            plt.legend()
            plt.show()

            # Anomalías en señales
            if df_test is not None:
                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                signals = ["f1", "f2", "f3"]
                for j, sig in enumerate(signals):
                    axes[j].plot(X_test.index, df_test[sig], label=sig)
                    axes[j].scatter(
                        X_test.index[y_test == 1],
                        df_test[sig][y_test == 1],
                        color="red", marker="x", label="Anomalías reales"
                    )
                    axes[j].scatter(
                        X_test.index[y_pred == 1],
                        df_test[sig][y_pred == 1],
                        facecolors="none", edgecolors="blue", label="Anomalías detectadas"
                    )
                    axes[j].set_title(f"Señal {sig}")
                    axes[j].legend(loc="upper right")

                plt.suptitle(f"Detección de anomalías - {model_class.__name__} {kwargs}")
                plt.tight_layout()
                plt.show()

    df_final = pd.DataFrame(final_results)

    # 4. Mostrar resultados en orden correcto
    if plot_results:
        print("Resultados de validación (grid search en train+val):")
        display(df_results.round(3))

        print("="*60)
        print(f"Top {top_k} configuraciones (ordenadas por {sort_metric} en validación):")
        print("="*60)
        display(topk)

        print("="*60)
        print("Resultados finales en TEST (top-5 configs):")
        print("="*60)
        display(df_final.round(3))

    return df_results, topk, df_final
