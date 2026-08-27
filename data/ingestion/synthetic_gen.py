"""
Synthetic High-Frequency Market Data Generator.

Generates realistic L2 order book depth (bids/asks), trade executions,
microstructure noise, and regime-switching volatility for offline research,
backtesting, and fallback modes without external API dependencies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class SyntheticMarketDataGenerator:
    """
    Simulates high-frequency L2 order book snapshots and trade executions with
    realistic microstructure properties:
    - Multi-level order book depth (bids and asks up to 10 levels)
    - Autocorrelated order flow imbalance (OBI dynamics)
    - Volatility regime switching (Low Vol vs High Vol)
    - Pareto-distributed trade sizes and aggressor side flags
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_orderbook_and_trades(
        self,
        n_steps: int = 5000,
        base_price: float = 50000.0,
        freq_ms: int = 100,
        n_levels: int = 10,
        start_time: str = "2026-08-27 00:00:00"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates aligned L2 orderbook snapshots and trade execution logs.

        Parameters
        ----------
        n_steps : int
            Number of time steps (ticks/bars).
        base_price : float
            Initial mid price (e.g. BTC/USDT price).
        freq_ms : int
            Time frequency in milliseconds per step.
        n_levels : int
            Number of order book depth levels (1 to 10).
        start_time : str
            Start timestamp ISO format.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (orderbook_df, trades_df)
        """
        timestamps = pd.date_range(start=start_time, periods=n_steps, freq=f"{freq_ms}ms")
        
        # Volatility regime switching process (Markov chain)
        # 0: Low Vol (0.0001 per step), 1: High Vol (0.0005 per step)
        regimes = np.zeros(n_steps, dtype=int)
        p_stay_low = 0.98
        p_stay_high = 0.95
        
        curr_regime = 0
        for t in range(1, n_steps):
            if curr_regime == 0:
                if np.random.rand() > p_stay_low:
                    curr_regime = 1
            else:
                if np.random.rand() > p_stay_high:
                    curr_regime = 0
            regimes[t] = curr_regime

        volatilities = np.where(regimes == 0, 0.0001, 0.0005)

        # Autocorrelated Order Imbalance state (AR(1) process)
        obi_state = np.zeros(n_steps)
        for t in range(1, n_steps):
            obi_state[t] = 0.7 * obi_state[t-1] + np.random.normal(0, 0.3)
        obi_state = np.clip(obi_state, -0.95, 0.95)

        # Mid price drift driven by OBI state + random shock
        price_returns = 0.5 * obi_state * volatilities + np.random.normal(0, 1, n_steps) * volatilities
        mid_prices = base_price * np.exp(np.cumsum(price_returns))

        # Build orderbook levels
        ob_records = []
        trade_records = []

        for t in range(n_steps):
            ts = timestamps[t]
            mid = mid_prices[t]
            vol = volatilities[t]
            obi = obi_state[t]

            # Dynamic bid-ask spread
            spread = max(0.5, mid * vol * (1.0 + np.random.exponential(0.5)))
            half_spread = spread / 2.0
            
            best_bid = mid - half_spread
            best_ask = mid + half_spread

            # Base volume at level 1 influenced by OBI
            base_vol = np.random.exponential(2.0) + 1.0
            bid_vol_l1 = base_vol * (1.0 + obi)
            ask_vol_l1 = base_vol * (1.0 - obi)

            row = {
                "timestamp": ts,
                "mid_price": mid,
                "spread": spread,
                "regime": regimes[t],
                "bid_price_1": round(best_bid, 2),
                "bid_qty_1": round(max(0.1, bid_vol_l1), 4),
                "ask_price_1": round(best_ask, 2),
                "ask_qty_1": round(max(0.1, ask_vol_l1), 4),
            }

            # Generate depth levels 2 to n_levels
            for lvl in range(2, n_levels + 1):
                step_dist = spread * (0.5 + 0.2 * (lvl - 1))
                b_p = round(best_bid - step_dist, 2)
                a_p = round(best_ask + step_dist, 2)
                
                # Volume decays or increases slightly with depth
                depth_factor = 1.0 + 0.15 * lvl
                b_q = round(max(0.1, base_vol * depth_factor * (1.0 + 0.8 * obi) * np.random.lognormal(0, 0.2)), 4)
                a_q = round(max(0.1, base_vol * depth_factor * (1.0 - 0.8 * obi) * np.random.lognormal(0, 0.2)), 4)

                row[f"bid_price_{lvl}"] = b_p
                row[f"bid_qty_{lvl}"] = b_q
                row[f"ask_price_{lvl}"] = a_p
                row[f"ask_qty_{lvl}"] = a_q

            ob_records.append(row)

            # Generate trade executions with probability proportional to volatility & trade intensity
            p_trade = 0.3 + 0.4 * (vol / 0.0005)
            if np.random.rand() < p_trade:
                # Aggressor side is biased by current OBI
                is_buy = np.random.rand() < (0.5 + 0.4 * obi)
                side = "BUY" if is_buy else "SELL"
                trade_price = best_ask if is_buy else best_bid
                trade_qty = round(float(np.random.pareto(a=2.5) * 0.5 + 0.05), 4)

                trade_records.append({
                    "timestamp": ts,
                    "price": trade_price,
                    "volume": trade_qty,
                    "side": side
                })

        ob_df = pd.DataFrame(ob_records)
        trades_df = pd.DataFrame(trade_records)

        return ob_df, trades_df


if __name__ == "__main__":
    gen = SyntheticMarketDataGenerator(seed=42)
    ob, trades = gen.generate_orderbook_and_trades(n_steps=1000)
    print(f"Generated {len(ob)} orderbook snapshots and {len(trades)} trades.")
    print("Orderbook sample:")
    print(ob[["timestamp", "mid_price", "bid_price_1", "bid_qty_1", "ask_price_1", "ask_qty_1"]].head())
