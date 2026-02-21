import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from src.ingestion.base_collector import BaseCollector
from src.utils.logger import log


class FEWSNETCollector(BaseCollector):
    """
    Collector for FEWS NET food insecurity data.
    IPC phases: 1=Minimal, 2=Stressed, 3=Crisis, 4=Emergency, 5=Famine
    """

    COUNTRY_CODES = {
        "CAR": "CF", "SEN": "SN", "MRT": "MR",
        "NER": "NE", "MLI": "ML", "TCD": "TD",
        "COD": "CD", "SSD": "SS", "ETH": "ET", "SOM": "SO"
    }

    def __init__(self):
        super().__init__("fewsnet")
        self.base_url = self.config["sources"]["fewsnet"]["base_url"]

    def collect(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect IPC food insecurity data for a country."""
        iso_code = self.COUNTRY_CODES.get(country_code, country_code)
        params = {
            "country": iso_code,
            "start_date": start_date,
            "end_date": end_date,
            "format": "json"
        }
        try:
            data = self._make_request(
                f"{self.base_url}/ipcpackage/",
                params=params
            )
            df = self._parse_ipc_data(data, country_code)
            filename = f"fewsnet_{country_code}_{start_date}_{end_date}.json"
            self.save_raw(data, filename)
            return df
        except Exception as e:
            log.warning(f"FEWS NET API unavailable, generating synthetic data for {country_code}")
            return self._generate_synthetic_data(country_code, start_date, end_date)

    def _parse_ipc_data(self, data: dict, country_code: str) -> pd.DataFrame:
        """Parse raw IPC data into DataFrame."""
        records = []
        for entry in data.get("results", []):
            records.append({
                "country": country_code,
                "date": entry.get("reference_date"),
                "ipc_phase": entry.get("overall_phase"),
                "population_affected": entry.get("population_in_need", 0),
                "source": "FEWS NET"
            })
        return pd.DataFrame(records)

    def _generate_synthetic_data(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic data for development/testing."""
        COUNTRY_PROFILES = {
            "SOM": {"base_ipc": 3.5, "volatility": 0.8, "pop_millions": 17},
            "SSD": {"base_ipc": 3.2, "volatility": 0.7, "pop_millions": 11},
            "ETH": {"base_ipc": 2.8, "volatility": 0.6, "pop_millions": 120},
            "CAR": {"base_ipc": 3.0, "volatility": 0.7, "pop_millions": 5},
            "NER": {"base_ipc": 2.5, "volatility": 0.5, "pop_millions": 25},
            "MLI": {"base_ipc": 2.3, "volatility": 0.5, "pop_millions": 22},
            "TCD": {"base_ipc": 2.6, "volatility": 0.6, "pop_millions": 17},
            "COD": {"base_ipc": 2.9, "volatility": 0.6, "pop_millions": 100},
            "MRT": {"base_ipc": 2.0, "volatility": 0.4, "pop_millions": 5},
            "SEN": {"base_ipc": 1.8, "volatility": 0.3, "pop_millions": 17},
        }

        profile = COUNTRY_PROFILES.get(country_code, {"base_ipc": 2.5, "volatility": 0.5, "pop_millions": 10})
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        np.random.seed(hash(country_code) % 2**32)

        ipc_phases = np.clip(
            profile["base_ipc"] + np.random.normal(0, profile["volatility"], len(dates)),
            1, 5
        )
        pop_affected = (ipc_phases / 5) * profile["pop_millions"] * 1e6 * (
            1 + np.random.normal(0, 0.1, len(dates))
        )

        df = pd.DataFrame({
            "country": country_code,
            "date": dates,
            "ipc_phase": ipc_phases.round(1),
            "population_affected": pop_affected.astype(int).clip(0),
            "source": "synthetic"
        })

        filename = f"fewsnet_{country_code}_synthetic.json"
        self.save_raw(df.to_dict(orient="records"), filename)
        return df


if __name__ == "__main__":
    collector = FEWSNETCollector()
    df = collector.collect("SOM", "2020-01-01", "2024-01-01")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"IPC range: {df['ipc_phase'].min():.1f} - {df['ipc_phase'].max():.1f}")