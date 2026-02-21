import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import log


class HumanitarianFeatureEngineer:
    """
    Feature engineering for humanitarian crisis forecasting.
    Creates lag features, rolling statistics, and composite indicators.
    """

    def __init__(self, forecast_horizon: int = 3):
        self.forecast_horizon = forecast_horizon
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        log.info("Starting feature engineering...")
        df = df.copy()
        df = df.sort_values(["country", "date"]).reset_index(drop=True)

        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_interaction_features(df)
        df = self._create_target_variable(df)
        df = self._create_temporal_features(df)
        df = df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in df.columns
                              if c not in ["country", "date", "target_crisis_severity"]]

        log.success(f"Feature engineering complete: {df.shape}")
        log.info(f"Features created: {len(self.feature_names)}")
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for each indicator."""
        lag_cols = ["ipc_phase", "total_events", "total_fatalities",
                    "total_displaced", "crisis_severity_index"]

        for col in lag_cols:
            if col not in df.columns:
                continue
            for lag in [1, 2, 3, 6]:
                feat_name = f"{col}_lag{lag}"
                df[feat_name] = df.groupby("country")[col].shift(lag)
                self.feature_names.append(feat_name)

        log.debug("Lag features created ✓")
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        roll_cols = ["ipc_phase", "total_events", "total_fatalities", "total_displaced"]

        for col in roll_cols:
            if col not in df.columns:
                continue
            for window in [3, 6]:
                grp = df.groupby("country")[col]

                df[f"{col}_roll{window}_mean"] = grp.transform(
                    lambda x: x.shift(1).rolling(window).mean()
                )
                df[f"{col}_roll{window}_std"] = grp.transform(
                    lambda x: x.shift(1).rolling(window).std()
                )
                df[f"{col}_roll{window}_max"] = grp.transform(
                    lambda x: x.shift(1).rolling(window).max()
                )

        log.debug("Rolling features created ✓")
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between crisis dimensions."""

        # Conflict x Food = Combined stress
        if "ipc_phase" in df.columns and "total_events" in df.columns:
            df["conflict_food_stress"] = df["ipc_phase"] * np.log1p(df["total_events"])

        # Displacement ratio
        if "total_displaced" in df.columns and "idps" in df.columns:
            df["displacement_intensity"] = np.log1p(df["total_displaced"])

        # Fatality rate per event
        if "total_fatalities" in df.columns and "total_events" in df.columns:
            df["fatality_rate"] = df["total_fatalities"] / (df["total_events"] + 1)

        # Crisis acceleration (month-over-month change)
        if "crisis_severity_index" in df.columns:
            df["crisis_acceleration"] = df.groupby("country")["crisis_severity_index"].diff()

        log.debug("Interaction features created ✓")
        return df

    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: future crisis severity."""
        if "crisis_severity_index" not in df.columns:
            # Recalcule si absent
            for col in ["ipc_phase", "total_events", "total_fatalities", "total_displaced"]:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val + 1e-8)

            norm_cols = [c for c in df.columns if c.endswith("_norm")]
            df["crisis_severity_index"] = df[norm_cols].mean(axis=1)

        df["target_crisis_severity"] = df.groupby("country")["crisis_severity_index"].shift(
            -self.forecast_horizon
        )

        log.debug(f"Target variable created (horizon={self.forecast_horizon} months) ✓")
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar features."""
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year

        # Saison des pluies / soudure (Afrique subsaharienne)
        df["lean_season"] = df["month"].isin([5, 6, 7, 8]).astype(int)
        df["harvest_season"] = df["month"].isin([10, 11, 12]).astype(int)

        # Encodage cyclique du mois
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Encodage pays
        country_codes = {c: i for i, c in enumerate(df["country"].unique())}
        df["country_code"] = df["country"].map(country_codes)

        log.debug("Temporal features created ✓")
        return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/dataset_with_csi.csv", parse_dates=["date"])
    engineer = HumanitarianFeatureEngineer(forecast_horizon=3)
    df_features = engineer.fit_transform(df)

    output_path = Path("data/processed/features_dataset.csv")
    df_features.to_csv(output_path, index=False)

    print(f"\n✅ Features dataset saved: {output_path}")
    print(f"   Shape: {df_features.shape}")
    print(f"   Features: {len(engineer.feature_names)}")
    print(f"\n📋 Feature list:\n{engineer.feature_names}")