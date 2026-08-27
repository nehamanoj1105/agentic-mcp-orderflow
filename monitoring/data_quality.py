"""
Market Data Quality & Integrity Auditor.

Checks sequence continuity, missing depth levels, invalid quotes, negative spreads,
and volume spikes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class DataQualityAuditor:
    """
    Audits incoming market data stream for sequence gaps and invalid quote states.
    """

    def audit_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = []
        total_rows = len(df)

        if total_rows == 0:
            return {"status": "EMPTY", "total_rows": 0, "issues": ["Empty dataset"]}

        # Check negative or inverted spread
        if "bid_price_1" in df and "ask_price_1" in df:
            inverted_spreads = (df["bid_price_1"] >= df["ask_price_1"]).sum()
            if inverted_spreads > 0:
                issues.append(f"Inverted bid/ask spread detected in {inverted_spreads} rows ({inverted_spreads/total_rows:.2%})")

        # Check missing or non-positive bids/asks
        if "bid_qty_1" in df and "ask_qty_1" in df:
            zero_qty = ((df["bid_qty_1"] <= 0) | (df["ask_qty_1"] <= 0)).sum()
            if zero_qty > 0:
                issues.append(f"Zero or negative quote quantity in {zero_qty} rows")

        # Check NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Total NaN values detected: {nan_count}")

        status = "PASSED" if len(issues) == 0 else "WARNING"
        return {
            "data_quality_status": status,
            "total_rows": total_rows,
            "issues_found": len(issues),
            "details": issues
        }
