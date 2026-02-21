import pandas as pd
from pathlib import Path
from src.ingestion.fewsnet_collector import FEWSNETCollector
from src.ingestion.acled_collector import ACLEDCollector
from src.ingestion.unhcr_collector import UNHCRCollector
from src.utils.logger import log


def run_ingestion_pipeline(start_date: str = "2018-01-01", end_date: str = "2024-01-01"):
    """
    Run the full data ingestion pipeline.
    Collects data from FEWS NET, ACLED, and UNHCR for all countries.
    """
    log.info("=" * 50)
    log.info("Starting HumaCrisis Data Ingestion Pipeline")
    log.info(f"Period: {start_date} → {end_date}")
    log.info("=" * 50)

    collectors = {
        "fewsnet": FEWSNETCollector(),
        "acled": ACLEDCollector(),
        "unhcr": UNHCRCollector(),
    }

    all_data = {}
    for name, collector in collectors.items():
        log.info(f"\n📥 Collecting {name.upper()} data...")
        all_data[name] = collector.collect_all_countries(start_date, end_date)

    log.info("\n🔗 Merging datasets...")
    merged = merge_datasets(all_data, start_date, end_date)

    output_path = Path("data/processed/merged_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    log.success(f"✅ Merged dataset saved: {output_path}")
    log.info(f"   Shape: {merged.shape}")
    log.info(f"   Countries: {merged['country'].nunique()}")
    log.info(f"   Date range: {merged['date'].min()} → {merged['date'].max()}")

    return merged


def merge_datasets(all_data: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Merge all data sources into a single DataFrame."""
    countries = ["CAR", "SEN", "MRT", "NER", "MLI", "TCD", "COD", "SSD", "ETH", "SOM"]
    frames = []

    for country in countries:
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        df = pd.DataFrame({"country": country, "date": dates})

        for source, country_data in all_data.items():
            if country_data.get(country) is not None:
                source_df = country_data[country].drop(columns=["source"], errors="ignore")
                df = df.merge(source_df, on=["country", "date"], how="left")

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["country", "date"]).reset_index(drop=True)
    return merged


if __name__ == "__main__":
    df = run_ingestion_pipeline()
    print("\n📊 Sample data:")
    print(df.head(20))