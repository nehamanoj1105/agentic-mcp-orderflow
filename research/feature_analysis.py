"""
Exploratory Feature Research & Quantitative Analysis.

Computes Pearson correlation, Spearman Information Coefficient (IC),
Mutual Information scores, distribution statistics, and saves research plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class FeatureAnalyzer:
    """
    Quantitative research toolkit for market microstructure feature evaluation.
    """

    def __init__(self, output_dir: str = "experiments/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        is_classification: bool = True
    ) -> pd.DataFrame:
        """
        Calculates Pearson correlation, Spearman IC, p-values, and Mutual Information.
        """
        records = []
        y = df[target_col].values

        for col in feature_cols:
            x = df[col].values

            # Remove NaNs or Infs
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 10:
                continue

            x_clean = x[mask]
            y_clean = y[mask]

            p_corr, p_val = pearsonr(x_clean, y_clean)
            s_ic, s_val = spearmanr(x_clean, y_clean)

            records.append({
                "feature": col,
                "pearson_corr": p_corr,
                "pearson_p_val": p_val,
                "spearman_ic": s_ic,
                "spearman_p_val": s_val,
                "std": np.std(x_clean),
                "skewness": float(pd.Series(x_clean).skew()),
                "kurtosis": float(pd.Series(x_clean).kurtosis())
            })

        results_df = pd.DataFrame(records).sort_values(by="spearman_ic", key=abs, ascending=False).reset_index(drop=True)

        # Compute Mutual Information if dataset is reasonably sized
        if len(df) > 50 and len(feature_cols) > 0:
            X_clean = df[feature_cols].fillna(0.0).values
            if is_classification:
                mi = mutual_info_classif(X_clean, y, random_state=42)
            else:
                mi = mutual_info_regression(X_clean, y, random_state=42)

            mi_map = dict(zip(feature_cols, mi))
            results_df["mutual_info"] = results_df["feature"].map(mi_map).fillna(0.0)

        return results_df

    def plot_correlation_matrix(self, df: pd.DataFrame, feature_cols: List[str], title: str = "Correlation Matrix") -> str:
        """
        Plots and saves feature correlation heatmap.
        """
        plt.figure(figsize=(10, 8))
        corr = df[feature_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "correlation_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_feature_vs_target(self, df: pd.DataFrame, feature_name: str, target_col: str) -> str:
        """
        Plots feature distribution vs prediction target.
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=feature_name, y=target_col, alpha=0.3)
        sns.regplot(data=df, x=feature_name, y=target_col, scatter=False, color="red")
        plt.title(f"Feature vs Future Return: {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel(target_col)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"feat_vs_target_{feature_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
