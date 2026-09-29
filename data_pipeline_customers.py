import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REAL_CREDIT_PARQUET, RANDOM_SEED, TARGET_COL
from logger import logger

def load_and_process_credit_portfolio(sample_size: int = None) -> pd.DataFrame:
    """
    Loads the real 30,000-customer credit card portfolio and engineers
    statistically verified retail banking risk features and delinquency metrics.
    """
    logger.info(f"Loading real credit card dataset from {REAL_CREDIT_PARQUET}")
    df = pd.read_parquet(REAL_CREDIT_PARQUET)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).copy()
    else:
        df = df.copy()

    # Standardize column naming
    col_mapping = {
        "ID": "customer_id",
        "LIMIT_BAL": "limit_bal",
        "SEX": "sex",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
        "AGE": "age",
        "default payment next month": TARGET_COL,
        "default.payment.next.month": TARGET_COL
    }
    df = df.rename(columns=col_mapping)
    
    # Harmonize repayment status columns (PAY_0 through PAY_6)
    pay_cols = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]
    
    # 1. Delinquency features: maximum delinquency and recent delinquency
    df["pay_max_delinquency"] = df[pay_cols].max(axis=1).clip(lower=0)
    df["pay_recent_delinquency"] = df["PAY_0"].clip(lower=0) if "PAY_0" in df.columns else df[pay_cols[0]].clip(lower=0)
    
    # 2. Billing & Payment aggregations
    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_amt_cols].mean(axis=1)
    
    # 3. Financial Burden Ratios
    # Revolving Utilization ratio (latest bill relative to credit limit, clamped to [0, 2])
    latest_bill = df[bill_cols[0]].clip(lower=0)
    df["revolving_utilization"] = (latest_bill / df["limit_bal"]).clip(0.0, 2.0)
    
    # Pay-to-Bill Ratio (Cash flow sufficiency)
    df["pay_to_bill_ratio"] = (df["avg_pay_amt"] / (df["avg_bill_amt"].abs() + 100.0)).clip(0.0, 5.0)
    
    # Bill to Limit Ratio
    df["bill_to_limit_ratio"] = (df["avg_bill_amt"] / df["limit_bal"]).clip(0.0, 3.0)
    
    # Standardize demographics
    df["education"] = df["education"].clip(1, 4)
    df["marriage"] = df["marriage"].clip(1, 3)
    
    # Add randomized causal treatment assignment {0: Control, 1: Limit Cut, 2: Payment Holiday}
    # for downstream CATE & ABM simulation
    rng = np.random.RandomState(RANDOM_SEED)
    df["treatment_group"] = rng.choice([0, 1, 2], size=len(df), p=[0.5, 0.25, 0.25])
    
    logger.info(f"Loaded {len(df)} customer records. Empirical default rate: {df[TARGET_COL].mean():.2%}")
    return df

if __name__ == "__main__":
    df = load_and_process_credit_portfolio()
    print("Processed Portfolio Summary:")
    print(df[["limit_bal", "age", "revolving_utilization", "pay_max_delinquency", TARGET_COL]].describe())
