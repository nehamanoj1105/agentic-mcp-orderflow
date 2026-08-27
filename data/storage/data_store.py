"""
Data Storage & Caching Layer.

Manages parquet persistence, train/test dataset retrieval, and in-memory caching.
"""

import os
import pandas as pd
from typing import Optional


class MarketDataStore:
    """
    Handles loading, saving, and partition caching for market data datasets.
    """

    def __init__(self, data_dir: str = "data_cache"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_dataset(self, df: pd.DataFrame, filename: str) -> str:
        """
        Saves DataFrame to Parquet format.
        """
        path = os.path.join(self.data_dir, filename)
        if not filename.endswith(".parquet"):
            path += ".parquet"
        df.to_parquet(path, index=False)
        return path

    def load_dataset(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Loads dataset from Parquet format.
        """
        path = os.path.join(self.data_dir, filename)
        if not filename.endswith(".parquet"):
            path += ".parquet"
        if not os.path.exists(path):
            return None
        return pd.read_parquet(path)
