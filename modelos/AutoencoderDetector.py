import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from modelos.base import BaseAnomalyDetector


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class AutoencoderDetector(BaseAnomalyDetector):
    def __init__(self, input_dim, latent_dim=8, lr=1e-3, epochs=50, batch_size=32,
                 use_scaler=True, device=None, verbose=True,
                 early_stopping=True, patience=5, delta=1e-4):
        """
        Detector de anomalías basado en Autoencoder (PyTorch).

        Parámetros
        ----------
        input_dim : int
            Número de variables de entrada.
        latent_dim : int, opcional (default=8)
            Dimensión del espacio latente.
        lr : float, opcional (default=1e-3)
            Learning rate para Adam.
        epochs : int, opcional (default=50)
            Número de épocas de entrenamiento por llamada a fit().
        batch_size : int, opcional (default=32)
            Tamaño de lote.
        use_scaler : bool, opcional (default=True)
            Si True, aplica normalización StandardScaler.
        device : str, opcional
            "cuda" o "cpu". Si None, detecta automáticamente.
        verbose : bool, opcional (default=True)
            Si True, muestra feedback durante entrenamiento.
        early_stopping : bool, opcional (default=True)
            Activa o desactiva el early stopping.
        patience : int, opcional (default=5)
            Número de épocas sin mejora para detener el entrenamiento.
        delta : float, opcional (default=1e-4)
            Mejora mínima que se considera relevante.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim, latent_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.threshold = None
        self.is_fitted = False
        self.verbose = verbose
        self.total_epochs_trained = 0  # entrenamiento incremental

        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta

    def preprocess(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler:
            if not self.is_fitted:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X

    def fit(self, X, X_val=None):
        """
        Entrena el autoencoder. 
        Si se llama varias veces, el entrenamiento continúa (incremental).
        Admite early stopping si se pasa un conjunto de validación.
        """
        X_proc = self.preprocess(X)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validación
        if X_val is not None:
            X_val_proc = self.preprocess(X_val)
            X_val_tensor = torch.tensor(X_val_proc, dtype=torch.float32).to(self.device)

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
                print(msg)

            # Early stopping
            if self.early_stopping and X_val is not None:
                if val_loss + self.delta < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if self.verbose:
                            print(f"⏹ Early stopping activado en epoch {self.total_epochs_trained}")
                        break

        # Calcular umbral como percentil 95 del error de reconstrucción en train
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
        self.threshold = np.percentile(errors, 95)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de predecir.")
        scores = self.anomaly_score(X)
        return (scores > self.threshold).astype(int)  # 1 = anómalo, 0 = normal

    def anomaly_score(self, X):
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de calcular scores.")
        X_proc = self.preprocess(X)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
        return errors
