import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models.xgboost_model import XGBoostCrisisForecaster
from src.models.lstm_model import LSTMCrisisForecaster
from src.utils.logger import log


class EnsembleCrisisForecaster:
    """
    Ensemble model combining XGBoost and LSTM predictions.
    Uses weighted averaging based on validation performance.
    """

    def __init__(self, forecast_horizon: int = 3):
        self.forecast_horizon = forecast_horizon
        self.xgb_model = XGBoostCrisisForecaster(forecast_horizon)
        self.lstm_model = LSTMCrisisForecaster(forecast_horizon)
        self.xgb_weight = 0.6
        self.lstm_weight = 0.4
        self.metrics = {}

    def train(self, df: pd.DataFrame):
        """Train both models and compute optimal weights."""
        log.info("=" * 50)
        log.info("Training Ensemble Crisis Forecaster")
        log.info("=" * 50)

        log.info("\n[1/2] Training XGBoost...")
        xgb_metrics = self.xgb_model.train(df)

        log.info("\n[2/2] Training LSTM...")
        lstm_metrics = self.lstm_model.train(df, epochs=50)

        # Compute weights inversely proportional to MAE
        xgb_mae = xgb_metrics["mae_mean"]
        lstm_mae = lstm_metrics["mae"]
        total = (1/xgb_mae) + (1/lstm_mae)
        self.xgb_weight = (1/xgb_mae) / total
        self.lstm_weight = (1/lstm_mae) / total

        log.info(f"\n📊 Ensemble Weights:")
        log.info(f"   XGBoost: {self.xgb_weight:.2%}")
        log.info(f"   LSTM:    {self.lstm_weight:.2%}")

        self.metrics = {
            "xgboost_mae": xgb_mae,
            "xgboost_r2": xgb_metrics["r2_mean"],
            "lstm_mae": lstm_mae,
            "lstm_r2": lstm_metrics["r2"],
            "xgb_weight": self.xgb_weight,
            "lstm_weight": self.lstm_weight
        }

        self._plot_model_comparison()
        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        xgb_preds = self.xgb_model.predict(df)
        return xgb_preds * self.xgb_weight

    def _plot_model_comparison(self):
        """Plot model comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0f1117")
        fig.suptitle("🎯 Model Comparison — XGBoost vs LSTM vs Ensemble",
                     color="white", fontsize=14)

        models = ["XGBoost", "LSTM"]
        maes = [self.metrics["xgboost_mae"], self.metrics["lstm_mae"]]
        r2s = [self.metrics["xgboost_r2"], self.metrics["lstm_r2"]]
        colors = ["#3498db", "#e74c3c"]

        for ax in axes:
            ax.set_facecolor("#0f1117")
            ax.tick_params(colors="white")

        bars1 = axes[0].bar(models, maes, color=colors, alpha=0.85, width=0.5)
        axes[0].set_title("MAE (lower is better)", color="white", pad=10)
        axes[0].set_ylabel("Mean Absolute Error", color="white")
        for bar, val in zip(bars1, maes):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", color="white", fontsize=12)

        bars2 = axes[1].bar(models, r2s, color=colors, alpha=0.85, width=0.5)
        axes[1].set_title("R² Score (higher is better)", color="white", pad=10)
        axes[1].set_ylabel("R² Score", color="white")
        for bar, val in zip(bars2, r2s):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", color="white", fontsize=12)

        plt.tight_layout()
        path = "reports/figures/08_model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        log.success(f"Saved: {path}")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features_dataset.csv", parse_dates=["date"])
    ensemble = EnsembleCrisisForecaster(forecast_horizon=3)
    metrics = ensemble.train(df)

    print("\n" + "=" * 50)
    print("✅ Ensemble Model Training Complete!")
    print("=" * 50)
    print(f"XGBoost → MAE: {metrics['xgboost_mae']:.4f} | R²: {metrics['xgboost_r2']:.4f}")
    print(f"LSTM    → MAE: {metrics['lstm_mae']:.4f} | R²: {metrics['lstm_r2']:.4f}")
    print(f"Weights → XGB: {metrics['xgb_weight']:.2%} | LSTM: {metrics['lstm_weight']:.2%}")