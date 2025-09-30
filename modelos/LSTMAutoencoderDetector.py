import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from modelos.base import BaseAnomalyDetector

class LSTMAutoEncoder(nn.Module):
    """
    LSTM Autoencoder para detección de anomalías en series temporales.

    Inspirado en:
    - Implementaciones de referencia en el repositorio TranAD (Tuli et al., 2022):
      https://github.com/imperial-qore/TranAD

    Parámetros
    ----------
    num_layers : int
        Número de capas LSTM en encoder y decoder.
    hidden_size : int
        Dimensión de la capa oculta LSTM.
    nb_feature : int
        Número de variables de entrada (features).
    dropout : float
        Dropout entre capas.
    device : torch.device
        Dispositivo de entrenamiento (CPU o GPU).
    """
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, nb_feature)
        )

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        out, _ = self.decoder(x, (h, c))  
        out = self.fc(out)
        return out

class LSTMAutoencoderDetector(BaseAnomalyDetector):
    def __init__(
        self,
        input_dim,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        epochs=50,
        batch_size=32,
        seq_len=20,
        use_scaler=True,
        device=None,
        verbose=False,
        early_stopping=True,
        patience=5,
        delta=1e-4
    ):
        """
        Detector de anomalías basado en LSTM Autoencoder.

        Parámetros
        ----------
        input_dim : int
            Dimensión de entrada (#features).
        hidden_size : int
            Número de unidades ocultas en cada capa LSTM.
        num_layers : int
            Número de capas LSTM en encoder y decoder.
        dropout : float
            Dropout entre capas LSTM.
        lr : float
            Learning rate para Adam.
        epochs : int
            Épocas por llamada a fit().
        batch_size : int
            Tamaño de lote.
        seq_len : int
            Longitud de las secuencias temporales usadas en entrenamiento.
        use_scaler : bool
            Si True, aplica StandardScaler.
        device : str
            "cuda" o "cpu".
        early_stopping : bool
            Si True, activa early stopping basado en validación.
        patience : int
            Número de épocas sin mejora antes de detener.
        delta : float
            Mínima mejora en validación para resetear paciencia.
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMAutoEncoder(num_layers, hidden_size, input_dim, dropout, self.device).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.threshold = None
        self.is_fitted = False
        self.verbose = verbose
        self.total_epochs_trained = 0

        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta

    def preprocess(self, X, retrain=True):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler:
            if retrain:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X

    def _create_sequences(self, X):
        """Convierte matriz [T, features] en secuencias [n_seq, seq_len, features]."""
        seqs = []
        for i in range(len(X) - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])
        return np.stack(seqs)

    def fit(self, X, X_val=None):
        X_seq = self._create_sequences(X)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validación
        if X_val is not None:
            X_val_seq = self._create_sequences(X_val)
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(self.device)

        best_val_loss = np.inf
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                xb = batch[0]
                self.optimizer.zero_grad()
                x_hat = self.model(xb)
                loss = self.criterion(x_hat, xb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            # Validación
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(X_val_tensor)
                    val_loss = self.criterion(recon, X_val_tensor).item()
                self.model.train()

            self.total_epochs_trained += 1
            if self.verbose:
                msg = f"[Epoch {self.total_epochs_trained}] Train Loss={epoch_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss={val_loss:.6f}"
                print(msg, end="\r")

            # Early stopping
            if self.early_stopping and X_val is not None:
                if val_loss + self.delta < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if self.verbose:
                            print(f"\n⏹ Early stopping activado en epoch {self.total_epochs_trained}")
                        break

        # Calcular threshold sobre errores de reconstrucción
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).cpu().numpy()
        self.threshold = np.percentile(errors, 95)

        self.is_fitted = True
        return self

    def predict(self, X):
        scores = self.anomaly_score(X)
        return (scores > self.threshold).astype(int)

    def anomaly_score(self, X):
        X_seq = self._create_sequences(X)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors_seq = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).cpu().numpy()

        # Expandir a longitud T: asignar score al último índice de cada secuencia
        scores = np.zeros(len(X_proc))
        scores[self.seq_len-1:] = errors_seq
        return scores

