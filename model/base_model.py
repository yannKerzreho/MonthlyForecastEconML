from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract wrapper for panel macro models.
    Any concrete model (RF, ESN, RNN, VAR, etc.)
    must inherit from this class.
    """

    def __init__(self, config: dict):
        """
        config: model-specific hyperparameters
        """
        self.config = config
        self.seed = None

    @abstractmethod
    def train_and_predict(self, data, horizon, train):
        """
        Predict horizon steps after trainnig if train is True.

        data is a dict of stationay time-series w.o. preprocessing (PCA, lags, ect)
        """
        pass