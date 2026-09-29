import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CUSTOMER_ONLY_FEATURES, MACRO_FEATURES, TARGET_COL
from logger import logger

def calculate_shannon_entropy(y: pd.Series) -> float:
    """Calculates base-2 Shannon Entropy H(Y) in bits."""
    probs = y.value_counts(normalize=True).values
    return float(entropy(probs, base=2))

def calculate_information_gain(feature: pd.Series, target: pd.Series, n_bins: int = 10) -> float:
    """
    Calculates Information Gain IG(Y, X) = H(Y) - H(Y|X).
    If feature is continuous, discretizes into quantile bins.
    """
    total_entropy = calculate_shannon_entropy(target)
    
    # Discretize continuous variables into bins
    if np.issubdtype(feature.dtype, np.number) and feature.nunique() > n_bins:
        try:
            binned_feature = pd.qcut(feature, q=n_bins, duplicates="drop")
        except Exception:
            binned_feature = pd.cut(feature, bins=n_bins)
    else:
        binned_feature = feature
        
    df = pd.DataFrame({"X": binned_feature, "Y": target}).dropna()
    n_total = len(df)
    
    # Calculate conditional entropy H(Y|X)
    cond_entropy = 0.0
    for val, group in df.groupby("X", observed=False):
        n_group = len(group)
        if n_group == 0:
            continue
        p_val = n_group / n_total
        group_entropy = calculate_shannon_entropy(group["Y"])
        cond_entropy += p_val * group_entropy
        
    info_gain = total_entropy - cond_entropy
    return max(0.0, float(info_gain))

def rank_features_by_information_gain(df: pd.DataFrame, feature_cols: list, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Ranks candidate risk features by Information Gain.
    """
    logger.info(f"Computing Information Gain for {len(feature_cols)} features...")
    ig_scores = {}
    for col in feature_cols:
        if col in df.columns:
            ig = calculate_information_gain(df[col], df[target_col])
            ig_scores[col] = ig
            
    rank_df = pd.DataFrame(list(ig_scores.items()), columns=["feature", "information_gain"])
    rank_df = rank_df.sort_values(by="information_gain", ascending=False).reset_index(drop=True)
    return rank_df

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Builds a leakage-safe ColumnTransformer for scaling numeric variables
    and encoding categorical variables.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

if __name__ == "__main__":
    from config import PROCESSED_MASTER_PARQUET
    df = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    all_features = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
    ranks = rank_features_by_information_gain(df, all_features)
    print("\nInformation Gain Feature Ranking:")
    print(ranks.to_string(index=False))
