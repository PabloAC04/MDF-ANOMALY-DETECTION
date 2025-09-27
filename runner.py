from utils import generate_synthetic_timeseries
from modelos.IsolationForest import IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Experiment:

    def __init__(self, model_class, model_kwargs=None):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}



    def run_experiment(self, X_train=None, X_val=None, X_test=None, y_val=None, y_test=None):

        # Instanciar modelo
        model = self.model_class(**(self.model_kwargs))
        model.fit(X_train.values if hasattr(X_train, "values") else X_train)

        # Validación
        y_pred_val = model.predict(X_val.values if hasattr(X_val, "values") else X_val)
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val)
        plt.title(f"Validación - {self.model_class.__name__}")
        plt.show()
        print(classification_report(y_val, y_pred_val))

        # Test
        y_pred_test = model.predict(X_test.values if hasattr(X_test, "values") else X_test)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
        plt.title(f"Test - {self.model_class.__name__}")
        plt.show()
        print(classification_report(y_test, y_pred_test))

if __name__ == "__main__":
    exp1 = Experiment(IsolationForest, {"n_trees": 100, "sample_size": 256, "contamination": 0.05})
    
    
