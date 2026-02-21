import pandas as pd
import numpy as np
from src.ingestion.base_collector import BaseCollector
from src.utils.logger import log


class UNHCRCollector(BaseCollector):
    """
    Collector for UNHCR displacement data.
    Covers: refugees, IDPs, returnees.
    """

    def __init__(self):
        super().__init__("unhcr")
        self.base_url = self.config["sources"]["unhcr"]["base_url"]

    def collect(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect displacement data for a country."""
        ISO3_CODES = {
            "CAR": "CAF", "SEN": "SEN", "MRT": "MRT",
            "NER": "NER", "MLI": "MLI", "TCD": "TCD",
            "COD": "COD", "SSD": "SSD", "ETH": "ETH", "SOM": "SOM"
        }

        iso3 = ISO3_CODES.get(country_code, country_code)
        url = f"{self.base_url}/population/?coa={iso3}&year=2020,2021,2022,2023,2024"

        try:
            data = self._make_request(url)
            df = self._parse_displacement_data(data, country_code, start_date, end_date)
            self.save_raw(data, f"unhcr_{country_code}_{start_date}_{end_date}.json")
            return df
        except Exception as e:
            log.warning(f"UNHCR API error: {e}, using synthetic data")
            return self._generate_synthetic_data(country_code, start_date, end_date)

    def _parse_displacement_data(self, data: dict, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Parse UNHCR annual data and interpolate to monthly."""
        items = data.get("items", [])
        if not items:
            return self._generate_synthetic_data(country_code, start_date, end_date)

        yearly = pd.DataFrame(items)
        yearly["date"] = pd.to_datetime(yearly["year"].astype(str))

        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        df = pd.DataFrame({"date": dates})
        df["country"] = country_code

        for col in ["refugees", "idps", "returnees"]:
            if col in yearly.columns:
                df[col] = np.interp(
                    df["date"].astype(int),
                    yearly["date"].astype(int),
                    pd.to_numeric(yearly[col], errors="coerce").fillna(0)
                )
            else:
                df[col] = 0

        df["total_displaced"] = df["refugees"] + df["idps"]
        df["source"] = "UNHCR"
        return df

    def _generate_synthetic_data(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic displacement data."""
        DISPLACEMENT_PROFILES = {
            "SOM": {"idps_millions": 2.9, "refugees_millions": 0.8, "trend": 0.02},
            "SSD": {"idps_millions": 1.8, "refugees_millions": 2.2, "trend": 0.01},
            "ETH": {"idps_millions": 3.5, "refugees_millions": 0.8, "trend": 0.03},
            "CAR": {"idps_millions": 0.6, "refugees_millions": 0.7, "trend": 0.01},
            "COD": {"idps_millions": 6.9, "refugees_millions": 0.5, "trend": 0.02},
            "MLI": {"idps_millions": 0.4, "refugees_millions": 0.1, "trend": 0.02},
            "NER": {"idps_millions": 0.3, "refugees_millions": 0.2, "trend": 0.01},
            "TCD": {"idps_millions": 0.4, "refugees_millions": 0.4, "trend": 0.01},
            "MRT": {"idps_millions": 0.0, "refugees_millions": 0.1, "trend": 0.00},
            "SEN": {"idps_millions": 0.0, "refugees_millions": 0.0, "trend": 0.00},
        }

        profile = DISPLACEMENT_PROFILES.get(country_code, {"idps_millions": 0.5, "refugees_millions": 0.2, "trend": 0.01})
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        n = len(dates)
        np.random.seed(hash(country_code) % 2**32 + 2)

        trend = np.linspace(1, 1 + profile["trend"] * n/12, n)
        noise = 1 + np.random.normal(0, 0.05, n)

        idps = (profile["idps_millions"] * 1e6 * trend * noise).clip(0).astype(int)
        refugees = (profile["refugees_millions"] * 1e6 * trend * noise).clip(0).astype(int)
        returnees = (idps * 0.05 * np.random.uniform(0.5, 1.5, n)).astype(int)

        df = pd.DataFrame({
            "country": country_code,
            "date": dates,
            "idps": idps,
            "refugees": refugees,
            "returnees": returnees,
            "total_displaced": idps + refugees,
            "source": "synthetic"
        })

        self.save_raw(df.to_dict(orient="records"), f"unhcr_{country_code}_synthetic.json")
        return df


if __name__ == "__main__":
    collector = UNHCRCollector()
    df = collector.collect("COD", "2020-01-01", "2024-01-01")
    print(df.head(10))
    print(f"\nMax displaced: {df['total_displaced'].max():,}")