import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingestion.fewsnet_collector import FEWSNETCollector
from src.ingestion.acled_collector import ACLEDCollector
from src.ingestion.unhcr_collector import UNHCRCollector
from src.features.feature_engineering import HumanitarianFeatureEngineer


class TestFEWSNETCollector:
    def setup_method(self):
        self.collector = FEWSNETCollector()

    def test_collect_returns_dataframe(self):
        df = self.collector.collect("SOM", "2022-01-01", "2023-01-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_collect_has_required_columns(self):
        df = self.collector.collect("SOM", "2022-01-01", "2023-01-01")
        assert "country" in df.columns
        assert "date" in df.columns
        assert "ipc_phase" in df.columns

    def test_ipc_phase_range(self):
        df = self.collector.collect("SOM", "2022-01-01", "2023-01-01")
        assert df["ipc_phase"].between(1, 5).all()

    def test_all_countries(self):
        countries = ["SOM", "ETH", "CAR"]
        for country in countries:
            df = self.collector.collect(country, "2022-01-01", "2023-01-01")
            assert len(df) > 0


class TestACLEDCollector:
    def setup_method(self):
        self.collector = ACLEDCollector()

    def test_collect_returns_dataframe(self):
        df = self.collector.collect("SOM", "2022-01-01", "2023-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_collect_has_conflict_columns(self):
        df = self.collector.collect("SOM", "2022-01-01", "2023-01-01")
        assert "total_events" in df.columns
        assert "total_fatalities" in df.columns

    def test_no_negative_values(self):
        df = self.collector.collect("SOM", "2022-01-01", "2023-01-01")
        assert (df["total_events"] >= 0).all()
        assert (df["total_fatalities"] >= 0).all()


class TestUNHCRCollector:
    def setup_method(self):
        self.collector = UNHCRCollector()

    def test_collect_returns_dataframe(self):
        df = self.collector.collect("COD", "2022-01-01", "2023-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_displacement_columns(self):
        df = self.collector.collect("COD", "2022-01-01", "2023-01-01")
        assert "idps" in df.columns
        assert "refugees" in df.columns
        assert "total_displaced" in df.columns


class TestFeatureEngineering:
    def setup_method(self):
        self.engineer = HumanitarianFeatureEngineer(forecast_horizon=3)

    def test_fit_transform_returns_dataframe(self):
        df = pd.read_csv("data/processed/dataset_with_csi.csv", parse_dates=["date"])
        result = self.engineer.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_feature_count(self):
        df = pd.read_csv("data/processed/dataset_with_csi.csv", parse_dates=["date"])
        result = self.engineer.fit_transform(df)
        assert len(self.engineer.feature_names) > 30

    def test_no_inf_values(self):
        df = pd.read_csv("data/processed/dataset_with_csi.csv", parse_dates=["date"])
        result = self.engineer.fit_transform(df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_target_variable_exists(self):
        df = pd.read_csv("data/processed/dataset_with_csi.csv", parse_dates=["date"])
        result = self.engineer.fit_transform(df)
        assert "target_crisis_severity" in result.columns