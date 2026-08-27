"""
Binance REST Client & Historical Data Downloader.

Fetches order book depth snapshots and recent trades from Binance public API.
Includes fallback to synthetic market data generator if network is offline or restricted.
"""

import logging
import requests
import pandas as pd
from typing import Tuple, Optional
from .synthetic_gen import SyntheticMarketDataGenerator

logger = logging.getLogger(__name__)


class BinanceRestClient:
    """
    Binance REST client for fetching live depth snapshots and recent trades.
    """

    def __init__(self, base_url: str = "https://api.binance.com", symbol: str = "BTCUSDT"):
        self.base_url = base_url
        self.symbol = symbol.upper()

    def get_order_book_snapshot(self, limit: int = 10) -> Optional[dict]:
        """
        Fetches public L2 order book snapshot for target symbol.
        """
        url = f"{self.base_url}/api/v3/depth"
        params = {"symbol": self.symbol, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Binance REST request failed for depth snapshot ({e}). Using synthetic fallback.")
            return None

    def get_recent_trades(self, limit: int = 500) -> Optional[list]:
        """
        Fetches recent public trade executions.
        """
        url = f"{self.base_url}/api/v3/trades"
        params = {"symbol": self.symbol, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Binance REST request failed for trades ({e}). Using synthetic fallback.")
            return None

    def get_market_dataset(
        self,
        n_steps: int = 2000,
        n_levels: int = 10,
        use_fallback: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves market dataset (orderbook snapshots and trade logs).
        If live request fails or offline mode requested, generates synthetic market data.
        """
        snapshot = self.get_order_book_snapshot(limit=n_levels)
        trades = self.get_recent_trades(limit=100)

        if snapshot is not None and trades is not None and len(snapshot.get("bids", [])) > 0:
            # Parse real Binance response into structured DataFrames
            timestamp = pd.Timestamp.now()
            bids = snapshot["bids"]
            asks = snapshot["asks"]
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2.0

            ob_record = {
                "timestamp": timestamp,
                "mid_price": mid_price,
                "spread": float(asks[0][0]) - float(bids[0][0]),
                "regime": 0
            }
            for i in range(min(n_levels, len(bids))):
                ob_record[f"bid_price_{i+1}"] = float(bids[i][0])
                ob_record[f"bid_qty_{i+1}"] = float(bids[i][1])
                ob_record[f"ask_price_{i+1}"] = float(asks[i][0])
                ob_record[f"ask_qty_{i+1}"] = float(asks[i][1])

            trade_records = []
            for t in trades:
                trade_records.append({
                    "timestamp": pd.to_datetime(t["time"], unit="ms"),
                    "price": float(t["price"]),
                    "volume": float(t["qty"]),
                    "side": "BUY" if t.get("isBuyerMaker", False) else "SELL"
                })

            ob_df = pd.DataFrame([ob_record])
            trades_df = pd.DataFrame(trade_records)

            if len(ob_df) < n_steps and use_fallback:
                logger.info("Real REST snapshot returned limited steps. Supplementing with synthetic data.")
                gen = SyntheticMarketDataGenerator()
                return gen.generate_orderbook_and_trades(n_steps=n_steps, base_price=mid_price, n_levels=n_levels)

            return ob_df, trades_df
        else:
            # Fallback to synthetic generator
            gen = SyntheticMarketDataGenerator()
            return gen.generate_orderbook_and_trades(n_steps=n_steps, n_levels=n_levels)
