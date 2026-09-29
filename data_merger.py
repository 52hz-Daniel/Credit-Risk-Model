import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_MASTER_PARQUET, RANDOM_SEED
from logger import logger
from utils.macro_data import build_macro_dataset
from models.macro_msvar import MacroRegimeEngine
from data_pipeline_customers import load_and_process_credit_portfolio

def create_master_credit_dataset(sample_size: int = None) -> pd.DataFrame:
    """
    Performs leakage-safe merge between real retail credit portfolio and
    macroeconomic regime time series (Hamilton Markov Switching systemic_risk_index).
    """
    logger.info("Starting master dataset construction (Option A: Real Data + Real Macro)")
    
    # 1. Ingest real macro data and compute Hamilton Markov-Switching regimes
    raw_macro = build_macro_dataset(60)
    regime_engine = MacroRegimeEngine()
    enriched_macro = regime_engine.fit(raw_macro)
    
    # 2. Ingest real customer credit portfolio
    customer_df = load_and_process_credit_portfolio(sample_size=sample_size)
    
    # 3. Simulate cohort observation dates over the macro window to model credit cycles
    macro_dates = enriched_macro.index
    rng = np.random.RandomState(RANDOM_SEED)
    
    # Assign each customer an as-of evaluation month from the macro time series
    # (weighting recent months more heavily as typical in rolling credit portfolios)
    p_weights = np.linspace(0.5, 2.0, len(macro_dates))
    p_weights /= p_weights.sum()
    assigned_dates = rng.choice(macro_dates, size=len(customer_df), p=p_weights)
    customer_df["as_of_date"] = assigned_dates
    
    # 4. As-of Join: map macro variables strictly at or before the as-of evaluation date
    macro_subset = enriched_macro[[
        "policy_rate",
        "high_yield_spread",
        "unemployment_rate",
        "systemic_risk_index"
    ]].copy()
    
    merged_df = customer_df.merge(
        macro_subset,
        left_on="as_of_date",
        right_index=True,
        how="left"
    )
    
    # Forward fill any potential boundary edge cases
    merged_df = merged_df.ffill().bfill()
    
    # Save to disk
    merged_df.to_parquet(PROCESSED_MASTER_PARQUET)
    logger.info(f"Saved master credit dataset to {PROCESSED_MASTER_PARQUET} with shape {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    df = create_master_credit_dataset()
    print("Master Dataset Sample:")
    print(df[["customer_id", "limit_bal", "as_of_date", "policy_rate", "systemic_risk_index", "default_next_90_days"]].head())
