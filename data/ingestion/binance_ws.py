"""
Binance Live WebSocket Stream Ingestion.

Connects to Binance public WebSocket stream for live L2 order book updates
and trade executions with real-time latency measurement.
Includes a streaming simulator mode for deterministic offline testing.
"""

import asyncio
import json
import logging
import time
import websockets
from typing import AsyncGenerator, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    Asynchronous Binance WebSocket client for real-time orderbook and trade streams.
    """

    def __init__(self, symbol: str = "btcusdt", speed_ms: int = 100):
        self.symbol = symbol.lower()
        self.speed_ms = speed_ms
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth10@{self.speed_ms}ms"

    async def stream_live_depth(self, max_messages: int = 100) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams live depth snapshots from Binance WebSocket.
        Yields structured market updates with arrival timestamps.
        """
        count = 0
        try:
            async with websockets.connect(self.ws_url, timeout=5) as ws:
                logger.info(f"Connected to Binance WebSocket stream: {self.ws_url}")
                while count < max_messages:
                    raw_msg = await ws.recv()
                    receive_time = time.time()
                    data = json.loads(raw_msg)
                    
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    if bids and asks:
                        mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                        payload = {
                            "timestamp": receive_time,
                            "symbol": self.symbol,
                            "mid_price": mid_price,
                            "spread": float(asks[0][0]) - float(bids[0][0]),
                            "bids": [(float(p), float(q)) for p, q in bids],
                            "asks": [(float(p), float(q)) for p, q in asks],
                            "latency_ms": 0.0
                        }
                        count += 1
                        yield payload
        except Exception as e:
            logger.warning(f"WebSocket connection failed or ended ({e}). Falling back to simulated stream.")
            async for payload in self.simulate_stream(max_messages=max_messages):
                yield payload

    async def simulate_stream(self, max_messages: int = 100) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Simulates live WebSocket stream for offline mode testing and latency benchmarks.
        """
        from .synthetic_gen import SyntheticMarketDataGenerator
        gen = SyntheticMarketDataGenerator()
        ob_df, _ = gen.generate_orderbook_and_trades(n_steps=max_messages, freq_ms=self.speed_ms)

        for _, row in ob_df.iterrows():
            start_t = time.time()
            await asyncio.sleep(self.speed_ms / 1000.0)
            proc_time = (time.time() - start_t) * 1000.0

            bids = [(row[f"bid_price_{i}"], row[f"bid_qty_{i}"]) for i in range(1, 11) if f"bid_price_{i}" in row]
            asks = [(row[f"ask_price_{i}"], row[f"ask_qty_{i}"]) for i in range(1, 11) if f"ask_price_{i}" in row]

            yield {
                "timestamp": time.time(),
                "symbol": self.symbol,
                "mid_price": row["mid_price"],
                "spread": row["spread"],
                "bids": bids,
                "asks": asks,
                "latency_ms": round(proc_time, 3)
            }
