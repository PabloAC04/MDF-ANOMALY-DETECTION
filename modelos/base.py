from abc import ABC, abstractmethod

class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X, y=None):
        pass

    @abstractmethod
    def anomaly_score(self, X, y=None):
        pass

    def preprocess(self, X, retrain=False):
        return X

