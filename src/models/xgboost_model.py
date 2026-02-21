import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.logger import log


class XGBoostCrisisForecaster:
    """
    XGBoost model for humanitarian crisis severity forecasting.
    Uses time-series cross-validation and SHAP explainability.
    """

    def __init__(self, forecast_horizon: int = 3):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.feature_names = []
        self.metrics = {}

        self.params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1
        }

    def prepare_data(self, df: pd.DataFrame):
        """Prepare features and target."""
        exclude = ["country", "date", "target_crisis_severity", "source"]
        exclude += [c for c in df.columns if c.endswith("_norm")]

        self.feature_names = [c for c in df.columns if c not in exclude]
        df_clean = df.dropna(subset=["target_crisis_severity"])

        X = df_clean[self.feature_names].fillna(0)
        y = df_clean["target_crisis_severity"]
        dates = df_clean["date"]
        countries = df_clean["country"]

        return X, y, dates, countries

    def train(self, df: pd.DataFrame):
        """Train with time-series cross-validation."""
        log.info("Training XGBoost Crisis Forecaster...")
        X, y, dates, countries = self.prepare_data(df)

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        mlflow.set_experiment("humacrisis_forecasting")

        with mlflow.start_run(run_name=f"xgboost_horizon{self.forecast_horizon}m"):
            mlflow.log_params(self.params)
            mlflow.log_param("forecast_horizon", self.forecast_horizon)
            mlflow.log_param("n_features", len(self.feature_names))

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBRegressor(**self.params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                preds = model.predict(X_val)
                mae = mean_absolute_error(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                r2 = r2_score(y_val, preds)
                cv_scores.append({"mae": mae, "rmse": rmse, "r2": r2})
                log.info(f"  Fold {fold+1} → MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

            # Train final model on all data
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y, verbose=False)

            # Metrics
            self.metrics = {
                "mae_mean": np.mean([s["mae"] for s in cv_scores]),
                "mae_std": np.std([s["mae"] for s in cv_scores]),
                "rmse_mean": np.mean([s["rmse"] for s in cv_scores]),
                "r2_mean": np.mean([s["r2"] for s in cv_scores]),
            }

            mlflow.log_metrics(self.metrics)
            mlflow.xgboost.log_model(self.model, "xgboost_model")

            log.success(f"XGBoost trained ✓")
            log.info(f"  CV MAE:  {self.metrics['mae_mean']:.4f} ± {self.metrics['mae_std']:.4f}")
            log.info(f"  CV RMSE: {self.metrics['rmse_mean']:.4f}")
            log.info(f"  CV R²:   {self.metrics['r2_mean']:.4f}")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        X, _, _, _ = self.prepare_data(df)
        return self.model.predict(X)

    def plot_feature_importance(self, top_n: int = 20):
        """Plot SHAP feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor("#0f1117")

        # XGBoost native importance
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=True).tail(top_n)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance)))
        importance.plot(kind="barh", ax=ax1, color=colors)
        ax1.set_facecolor("#0f1117")
        ax1.set_title(f"Top {top_n} Feature Importances", color="white", pad=12)
        ax1.tick_params(colors="white")
        ax1.set_xlabel("Importance Score", color="white")

        # Predictions vs Actuals (ax2)
        ax2.set_facecolor("#0f1117")
        ax2.set_title("Model Performance Summary", color="white", pad=12)
        metrics_text = (
            f"Cross-Validation Results (5 folds)\n\n"
            f"MAE:  {self.metrics['mae_mean']:.4f} ± {self.metrics['mae_std']:.4f}\n"
            f"RMSE: {self.metrics['rmse_mean']:.4f}\n"
            f"R²:   {self.metrics['r2_mean']:.4f}\n\n"
            f"Features: {len(self.feature_names)}\n"
            f"Horizon:  {self.forecast_horizon} months"
        )
        ax2.text(0.5, 0.5, metrics_text, transform=ax2.transAxes,
                fontsize=14, color="white", ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))
        ax2.axis("off")

        plt.tight_layout()
        path = "reports/figures/06_xgboost_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        log.success(f"Saved: {path}")
        plt.show()

    def save(self, path: str = "models/trained/xgboost_model.json"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        log.success(f"Model saved: {path}")

    def load(self, path: str = "models/trained/xgboost_model.json"):
        """Load model from disk."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        log.success(f"Model loaded: {path}")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/features_dataset.csv", parse_dates=["date"])
    forecaster = XGBoostCrisisForecaster(forecast_horizon=3)
    metrics = forecaster.train(df)
    forecaster.plot_feature_importance()
    forecaster.save()
    print(f"\n✅ XGBoost model trained and saved!")
    print(f"   MAE: {metrics['mae_mean']:.4f}")
    print(f"   R²:  {metrics['r2_mean']:.4f}")