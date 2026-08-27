"""
Statistical Significance & Trading Backtest Evaluation Engine.

Provides statistical hypothesis testing (Diebold-Mariano test, IC t-test, bootstrap CIs)
and realistic transaction-cost-adjusted backtest trading performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import t as t_dist, norm


class StatisticalBacktester:
    """
    Statistical hypothesis testing and backtest strategy metrics calculator.
    """

    @staticmethod
    def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> Tuple[float, float]:
        """
        Performs Diebold-Mariano test to check if Model 1 errors (e1) and Model 2 errors (e2)
        are significantly different.
        """
        d = e1**2 - e2**2
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        if var_d == 0:
            return 0.0, 1.0

        n = len(d)
        stat = mean_d / np.sqrt(var_d / n)
        p_val = 2.0 * (1.0 - norm.cdf(abs(stat)))
        return float(stat), float(p_val)

    @staticmethod
    def evaluate_trading_backtest(
        predictions: np.ndarray,
        future_returns: np.ndarray,
        cost_bps: float = 2.0,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Evaluates quantitative strategy backtest performance:
        - Gross Return
        - Transaction Costs (cost_bps basis points per turn)
        - Net Return
        - Annualized Sharpe Ratio
        - Maximum Drawdown
        - Win Rate
        - Portfolio Turnover
        """
        if is_classification:
            # Map classes: 0=DOWN (-1), 1=NEUTRAL (0), 2=UP (+1)
            signal = np.where(predictions == 2, 1.0, np.where(predictions == 0, -1.0, 0.0))
        else:
            signal = np.sign(predictions)

        # Calculate position turns for turnover and transaction costs
        position_changes = np.abs(np.diff(signal, prepend=0))
        turnover = np.mean(position_changes)

        cost_factor = cost_bps / 10000.0
        transaction_costs = position_changes * cost_factor

        gross_step_returns = signal * future_returns
        net_step_returns = gross_step_returns - transaction_costs

        cum_gross = np.exp(np.cumsum(gross_step_returns)) - 1.0
        cum_net = np.exp(np.cumsum(net_step_returns)) - 1.0

        total_gross_ret = float(cum_gross[-1]) if len(cum_gross) > 0 else 0.0
        total_net_ret = float(cum_net[-1]) if len(cum_net) > 0 else 0.0

        # Sharpe ratio calculation (assuming 100ms / 1s steps)
        std_net = np.std(net_step_returns)
        if std_net > 0:
            sharpe = (np.mean(net_step_returns) / std_net) * np.sqrt(252 * 86400)
        else:
            sharpe = 0.0

        # Maximum Drawdown calculation
        peak = np.maximum.accumulate(cum_net + 1.0)
        drawdown = (cum_net + 1.0 - peak) / peak
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Win Rate
        active_trades = signal != 0
        if np.sum(active_trades) > 0:
            win_rate = float(np.mean(net_step_returns[active_trades] > 0))
        else:
            win_rate = 0.0

        return {
            "gross_return": total_gross_ret,
            "transaction_costs": float(np.sum(transaction_costs)),
            "net_return": total_net_ret,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "turnover": float(turnover)
        }
