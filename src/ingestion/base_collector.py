import requests
import yaml
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import log

load_dotenv()

class BaseCollector(ABC):
    """Base class for all data collectors."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.config = self._load_config()
        self.raw_path = Path(self.config["data"]["raw_path"]) / source_name
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "HumaCrisis-Forecasting/1.0",
            "Accept": "application/json"
        })
        log.info(f"Initialized {source_name} collector")

    def _load_config(self) -> dict:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

    def _make_request(self, url: str, params: dict = None, retries: int = 3) -> dict:
        """Make HTTP request with retry logic."""
        for attempt in range(retries):
            try:
                log.debug(f"Requesting: {url} | params: {params}")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                log.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    log.error(f"All {retries} attempts failed for {url}")
                    raise

    def save_raw(self, data, filename: str):
        """Save raw data to file."""
        import json
        filepath = self.raw_path / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.success(f"Saved: {filepath}")
        return filepath

    @abstractmethod
    def collect(self, country_code: str, start_date: str, end_date: str):
        """Collect data for a specific country and date range."""
        pass

    def collect_all_countries(self, start_date: str, end_date: str):
        """Collect data for all configured countries."""
        countries = self.config["data"]["countries"]
        results = {}
        for country in countries:
            log.info(f"Collecting {self.source_name} data for {country}...")
            try:
                results[country] = self.collect(country, start_date, end_date)
                log.success(f"{country} done ✓")
            except Exception as e:
                log.error(f"{country} failed: {e}")
                results[country] = None
        return results