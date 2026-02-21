import pandas as pd
import numpy as np
import os
from src.ingestion.base_collector import BaseCollector
from src.utils.logger import log


class ACLEDCollector(BaseCollector):
    """
    Collector for ACLED conflict data.
    Covers: battles, explosions, violence against civilians, protests.
    """

    def __init__(self):
        super().__init__("acled")
        self.base_url = self.config["sources"]["acled"]["base_url"]
        self.api_key = os.getenv("ACLED_API_KEY")
        self.email = os.getenv("ACLED_EMAIL")

    def collect(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect conflict events for a country."""
        COUNTRY_NAMES = {
            "CAR": "Central African Republic", "SEN": "Senegal",
            "MRT": "Mauritania", "NER": "Niger", "MLI": "Mali",
            "TCD": "Chad", "COD": "Democratic Republic of Congo",
            "SSD": "South Sudan", "ETH": "Ethiopia", "SOM": "Somalia"
        }

        if not self.api_key or self.api_key == "your_acled_api_key":
            log.warning(f"No ACLED API key, generating synthetic data for {country_code}")
            return self._generate_synthetic_data(country_code, start_date, end_date)

        params = {
            "key": self.api_key,
            "email": self.email,
            "country": COUNTRY_NAMES.get(country_code, country_code),
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
            "fields": "event_date|event_type|fatalities|latitude|longitude|location",
            "limit": 5000,
            "format": "json"
        }

        try:
            data = self._make_request(self.base_url, params=params)
            df = self._parse_conflict_data(data, country_code)
            self.save_raw(data, f"acled_{country_code}_{start_date}_{end_date}.json")
            return df
        except Exception as e:
            log.warning(f"ACLED API error: {e}, using synthetic data")
            return self._generate_synthetic_data(country_code, start_date, end_date)

    def _parse_conflict_data(self, data: dict, country_code: str) -> pd.DataFrame:
        """Parse ACLED response into aggregated monthly DataFrame."""
        records = data.get("data", [])
        if not records:
            return pd.DataFrame()

        df_raw = pd.DataFrame(records)
        df_raw["event_date"] = pd.to_datetime(df_raw["event_date"])
        df_raw["fatalities"] = pd.to_numeric(df_raw["fatalities"], errors="coerce").fillna(0)
        df_raw["month"] = df_raw["event_date"].dt.to_period("M")

        df_monthly = df_raw.groupby("month").agg(
            total_events=("event_date", "count"),
            total_fatalities=("fatalities", "sum"),
            battles=("event_type", lambda x: (x == "Battles").sum()),
            violence_civilians=("event_type", lambda x: (x == "Violence against civilians").sum()),
        ).reset_index()

        df_monthly["country"] = country_code
        df_monthly["date"] = df_monthly["month"].dt.to_timestamp()
        df_monthly["source"] = "ACLED"
        return df_monthly.drop("month", axis=1)

    def _generate_synthetic_data(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic conflict data."""
        CONFLICT_PROFILES = {
            "SOM": {"base_events": 80,  "fatality_rate": 5.0, "volatility": 25},
            "SSD": {"base_events": 60,  "fatality_rate": 4.5, "volatility": 20},
            "ETH": {"base_events": 50,  "fatality_rate": 3.5, "volatility": 20},
            "CAR": {"base_events": 45,  "fatality_rate": 3.0, "volatility": 15},
            "MLI": {"base_events": 40,  "fatality_rate": 2.5, "volatility": 15},
            "NER": {"base_events": 30,  "fatality_rate": 2.0, "volatility": 12},
            "TCD": {"base_events": 35,  "fatality_rate": 2.5, "volatility": 12},
            "COD": {"base_events": 55,  "fatality_rate": 3.5, "volatility": 18},
            "MRT": {"base_events": 10,  "fatality_rate": 1.0, "volatility": 5},
            "SEN": {"base_events": 8,   "fatality_rate": 0.5, "volatility": 4},
        }

        profile = CONFLICT_PROFILES.get(country_code, {"base_events": 30, "fatality_rate": 2.0, "volatility": 10})
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        np.random.seed(hash(country_code) % 2**32 + 1)

        events = np.clip(
            profile["base_events"] + np.random.normal(0, profile["volatility"], len(dates)),
            0, None
        ).astype(int)

        fatalities = (events * profile["fatality_rate"] * (
            1 + np.random.normal(0, 0.3, len(dates))
        )).clip(0).astype(int)

        df = pd.DataFrame({
            "country": country_code,
            "date": dates,
            "total_events": events,
            "total_fatalities": fatalities,
            "battles": (events * 0.4).astype(int),
            "violence_civilians": (events * 0.3).astype(int),
            "source": "synthetic"
        })

        self.save_raw(df.to_dict(orient="records"), f"acled_{country_code}_synthetic.json")
        return df


if __name__ == "__main__":
    collector = ACLEDCollector()
    df = collector.collect("SOM", "2020-01-01", "2024-01-01")
    print(df.head(10))
    print(f"\nTotal fatalities: {df['total_fatalities'].sum():,}")