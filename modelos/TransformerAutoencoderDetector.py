import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cudf
from spot import SPOT

from modelos.base import BaseAnomalyDetector


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [L, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SimpleEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(SimpleEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(True)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + src2
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + src2
        return src


class SimpleDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(SimpleDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(True)

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + tgt2
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + tgt2
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = tgt + tgt2
        return tgt


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        d_model = 2 * input_dim  # como TranAD
        nhead = min(input_dim, d_model)  # nheads ≤ d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)

        # Encoder (1 capa)
        self.encoder = SimpleEncoderLayer(d_model, nhead, dim_feedforward=16, dropout=dropout)
        # Decoder (1 capa)
        self.decoder = SimpleDecoderLayer(d_model, nhead, dim_feedforward=16, dropout=dropout)

        # Salida: solo último timestep
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x_emb = self.input_proj(x)
        x_emb = self.pos_encoder(x_emb)

        memory = self.encoder(x_emb)
        out = self.decoder(x_emb, memory)

        # Solo el último timestep reconstruido
        x_hat = self.output_layer(out[:, -1, :])  # [batch, input_dim]
        return x_hat


# === Detector simplificado ===
class TransformerAutoencoderDetector(BaseAnomalyDetector):
    def __init__(self, lr=1e-3, epochs=50, batch_size=128, seq_len=10,
                 use_scaler=True, device=None, verbose=False):

        torch.backends.cudnn.benchmark = True
        self.input_dim = None
        self.seq_len = int(seq_len)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.use_scaler = use_scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if use_scaler else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.threshold = None
        self.is_fitted = False

    def preprocess(self, X, retrain=True):
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            X = X.to_numpy(dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        if self.scaler:
            X = self.scaler.fit_transform(X) if retrain else self.scaler.transform(X)

        return create_windows(X, self.seq_len)

    def preprocess_labels(self, y):
        if isinstance(y, (cudf.Series, cudf.DataFrame)):
            y = y.to_numpy()
        elif hasattr(y, "values"):  # pandas Series
            y = y.values
        else:
            y = np.asarray(y)
        return y[self.seq_len - 1:]

    def fit(self, X, X_val=None):
        if self.input_dim is None:
            self.input_dim = X.shape[2]
            self.seq_len = X.shape[1]
            self.model = TransformerAutoencoder(self.input_dim, self.seq_len).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, fused=(self.device.type == "cuda"))

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = np.inf
        epochs_no_improve = 0
        scaler = torch.amp.GradScaler(device=self.device)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            for xb, in loader:
                xb = xb.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type):
                    x_hat = self.model(xb)
                    feat_idx = torch.randperm(xb.size(-1))[: xb.size(-1) // 2]
                    loss = self.criterion(x_hat[:, feat_idx], xb[:, -1, feat_idx])

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            if self.verbose:
                print(f"[Epoch {epoch}] Train Loss={epoch_loss:.6f}", end="\r", flush=True)

        # Umbral sobre error de reconstrucción
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor[:, -1, :] - recon) ** 2, dim=1)
        
        # ===== Umbral con SPOT (POT) =====

        # split: 80% inicialización, 20% stream
        n_init = int(0.8 * len(errors))
        init_data, stream_data = errors[:n_init], errors[n_init:]

        s = SPOT(q=1e-4)  # nivel de riesgo
        s.fit(init_data, stream_data)
        s.initialize(level=0.98, verbose=False)
        results = s.run(dynamic=False)

        # threshold final = último umbral calculado
        self.threshold = results["thresholds"][-1]
        print(f"Threshold calculado con SPOT: {self.threshold:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X, y=None):
        scores = self.anomaly_score(X)
        preds = (scores > self.threshold).astype(int)
        if y is not None:
            y = self.preprocess_labels(y)
            return preds, y
        return preds

    def anomaly_score(self, X, y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((X_tensor[:, -1, :] - recon) ** 2, dim=1).cpu().numpy()
        if y is not None:
            y = self.preprocess_labels(y)
            return errors, y
        return errors


# === Utilidad: generar ventanas ===
def create_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = X.shape
    return np.stack([X[i:i + seq_len] for i in range(N - seq_len + 1)])
