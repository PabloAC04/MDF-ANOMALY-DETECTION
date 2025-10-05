import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import cupy as cp
import cudf
from spot import SPOT   


from modelos.base import BaseAnomalyDetector


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=16):
        super().__init__()
        # Encoder súper ligero
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder súper ligero
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim, input_dim)  # sin sigmoid
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector(BaseAnomalyDetector):
    def __init__(self, latent_dim=8, lr=1e-3, epochs=50, batch_size=32,
                 use_scaler=True, device=None, verbose=False,
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
        torch.backends.cudnn.benchmark = True
        self.input_dim = None
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.use_scaler = use_scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if use_scaler else None

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None

        self.threshold = None
        self.is_fitted = False
        self.verbose = verbose
        self.total_epochs_trained = 0  # entrenamiento incremental

        best_state = None

        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta

    def preprocess(self, X, retrain=True):
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            X = X.to_numpy(dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        if self.scaler:
            if retrain:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X

    def fit(self, X, X_val=None):
        """
        Entrena el autoencoder con AMP (Mixed Precision).
        Si se llama varias veces, el entrenamiento continúa (incremental).
        Admite early stopping si se pasa un conjunto de validación.
        """
        if self.input_dim is None:
            self.input_dim = int(X.shape[1])
            self.model = Autoencoder(self.input_dim, self.latent_dim).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, fused=(self.device.type == "cuda"))

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Validación
        if X_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        best_val_loss = np.inf
        epochs_no_improve = 0

        # ✅ Mixed Precision
        scaler = torch.amp.GradScaler()

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                xb = batch[0]

                xb = xb.contiguous()

                self.optimizer.zero_grad()

                # Forward con AMP
                with torch.amp.autocast(device_type=self.device.type):
                    x_hat = self.model(xb)
                    loss = self.criterion(x_hat, xb)

                # Backward con escala de gradiente
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()


                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            # Validación
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
                    recon = self.model(X_val_tensor)
                    val_loss = self.criterion(recon, X_val_tensor).item()
                self.model.train()

            self.total_epochs_trained += 1

            if self.verbose:
                msg = f"[Epoch {self.total_epochs_trained}] Train Loss={epoch_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss={val_loss:.6f}"
                print(msg, end="\r", flush=True)

            # Early stopping
            if self.early_stopping and X_val is not None:
                if val_loss + self.delta < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                        if self.verbose:
                            print(f"⏹ Early stopping activado en epoch {self.total_epochs_trained}")
                        break

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()

        # ===== Umbral con SPOT (estilo TranAD) =====

        # Usa solo errores de entrenamiento para calibrar
        train_errors = errors[:int(0.8 * len(errors))]
        test_errors = errors[int(0.8 * len(errors)):]

        q = 1e-3
        lm = [0.995, 1.4]  # nivel inicial y factor de amplificación

        while True:
            try:
                s = SPOT(q)
                s.fit(train_errors, test_errors)
                s.initialize(level=lm[0], min_extrema=False, verbose=False)
            except:
                lm[0] *= 0.999
            else:
                break

        ret = s.run(dynamic=False)
        thr_seq = np.array(ret["thresholds"], dtype=float)

        if len(thr_seq) > 0:
            base_thr = np.mean(thr_seq) * lm[1]
            perc_thr = np.percentile(train_errors, 99.7)
            self.threshold = float(max(base_thr, perc_thr))
        else:
            self.threshold = float(np.percentile(train_errors, 99.7))

        self.is_fitted = True
        return self


    def predict(self, X, y=None):
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de predecir.")
        scores = self.anomaly_score(X)
        return (scores > self.threshold).astype(int)  # 1 = anómalo, 0 = normal

    def anomaly_score(self, X, y=None):
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de calcular scores.")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
        return errors
