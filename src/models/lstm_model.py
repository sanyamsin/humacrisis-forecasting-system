import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.logger import log


class CrisisDataset(Dataset):
    """PyTorch Dataset for crisis time series."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 6):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.seq_len],
            self.y[idx + self.seq_len]
        )


class LSTMCrisisNet(nn.Module):
    """LSTM neural network for crisis forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out.squeeze()


class LSTMCrisisForecaster:
    """LSTM-based humanitarian crisis forecaster."""

    def __init__(self, forecast_horizon: int = 3, seq_len: int = 6):
        self.forecast_horizon = forecast_horizon
        self.seq_len = seq_len
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

    def prepare_data(self, df: pd.DataFrame):
        """Prepare sequences for LSTM."""
        exclude = ["country", "date", "target_crisis_severity", "source"]
        exclude += [c for c in df.columns if c.endswith("_norm")]
        feature_cols = [c for c in df.columns if c not in exclude]

        df_clean = df.dropna(subset=["target_crisis_severity"])
        X = df_clean[feature_cols].fillna(0).values
        y = df_clean["target_crisis_severity"].values

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, feature_cols

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train LSTM model."""
        log.info("Training LSTM Crisis Forecaster...")
        X, y, feature_cols = self.prepare_data(df)

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_dataset = CrisisDataset(X_train, y_train, self.seq_len)
        val_dataset = CrisisDataset(X_val, y_val, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model = LSTMCrisisNet(
            input_size=X.shape[1],
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion = nn.MSELoss()

        train_losses, val_losses = [], []
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    preds = self.model(X_batch)
                    val_loss += criterion(preds, y_batch).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("models/trained/lstm_best.pt")

            if (epoch + 1) % 10 == 0:
                log.info(f"  Epoch {epoch+1}/{epochs} → Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Final metrics on validation set
        val_preds, val_true = [], []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        self.metrics = {
            "mae": mean_absolute_error(val_true, val_preds),
            "rmse": np.sqrt(mean_squared_error(val_true, val_preds)),
            "r2": r2_score(val_true, val_preds)
        }

        log.success(f"LSTM trained ✓")
        log.info(f"  MAE:  {self.metrics['mae']:.4f}")
        log.info(f"  RMSE: {self.metrics['rmse']:.4f}")
        log.info(f"  R²:   {self.metrics['r2']:.4f}")

        self._plot_training(train_losses, val_losses)
        return self.metrics

    def _plot_training(self, train_losses, val_losses):
        """Plot training curves."""
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        ax.plot(train_losses, color="#3498db", label="Train Loss", linewidth=2)
        ax.plot(val_losses, color="#e74c3c", label="Val Loss", linewidth=2)
        ax.set_title("LSTM Training Curves", color="white", pad=12)
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("MSE Loss", color="white")
        ax.tick_params(colors="white")
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = "reports/figures/07_lstm_training.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        log.success(f"Saved: {path}")
        plt.show()

    def save(self, path: str = "models/trained/lstm_model.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "models/trained/lstm_model.pt", input_size: int = 71):
        self.model = LSTMCrisisNet(input_size=input_size).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features_dataset.csv", parse_dates=["date"])
    forecaster = LSTMCrisisForecaster(forecast_horizon=3)
    metrics = forecaster.train(df, epochs=50)
    print(f"\n✅ LSTM trained!")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R²:  {metrics['r2']:.4f}")