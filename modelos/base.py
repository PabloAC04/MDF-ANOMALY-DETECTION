from abc import ABC, abstractmethod

class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def anomaly_score(self, X):
        pass

    def preprocess(self, X, retrain=False):
        return X

