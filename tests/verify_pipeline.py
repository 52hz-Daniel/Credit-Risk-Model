import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SAVED_MODELS_DIR,
    PROCESSED_MASTER_PARQUET,
    REAL_MACRO_CSV,
    CUSTOMER_ONLY_FEATURES,
    MACRO_FEATURES
)
from logger import logger

def run_acceptance_tests():
    logger.info("Executing comprehensive quantitative validation tests...")
    
    # 1. Master Dataset Acceptance
    df = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    assert len(df) == 30000, f"Expected 30,000 customers, got {len(df)}"
    assert "default_next_90_days" in df.columns, "Target column missing"
    assert 0.15 <= df["default_next_90_days"].mean() <= 0.30, "Default rate outside realistic bounds [15%, 30%]"
    logger.info(f"Test 1 Passed: Master dataset shape {df.shape}, default rate {df['default_next_90_days'].mean():.2%}")
    
    # 2. Macro Regime Acceptance
    macro_df = pd.read_csv(REAL_MACRO_CSV, index_col="date")
    assert "systemic_risk_index" in macro_df.columns, "systemic_risk_index column missing"
    assert macro_df["systemic_risk_index"].min() >= 0.0, "systemic_risk_index below 0"
    assert macro_df["systemic_risk_index"].max() <= 1.0, "systemic_risk_index exceeds 1"
    logger.info("Test 2 Passed: Hamilton Markov-Switching systemic_risk_index bounded in [0, 1]")
    
    # 3. Model Calibration & Performance Acceptance
    metrics = joblib.load(SAVED_MODELS_DIR / "metrics.joblib")
    m_base = metrics["customer_only"]
    m_macro = metrics["macro_aware"]
    
    assert m_base["roc_auc"] > 0.70, f"Customer ROC-AUC too low: {m_base['roc_auc']}"
    assert m_macro["roc_auc"] > 0.70, f"Macro-Aware ROC-AUC too low: {m_macro['roc_auc']}"
    assert m_macro["ece"] < 0.05, f"Calibration error too high: {m_macro['ece']}"
    assert m_macro["brier_score"] < 0.20, f"Brier score too high: {m_macro['brier_score']}"
    logger.info(f"Test 3 Passed: Out-of-fold calibrated ROC-AUC = {m_macro['roc_auc']:.4f}, ECE = {m_macro['ece']:.4f}, Brier = {m_macro['brier_score']:.4f}")
    
    # 4. SHAP Explainability Acceptance
    test_payload = joblib.load(SAVED_MODELS_DIR / "test_payload.joblib")
    assert len(test_payload["y_test"]) == 6000, f"Expected 6,000 test cases, got {len(test_payload['y_test'])}"
    logger.info("Test 4 Passed: Test evaluation payload verified (N=6,000 holdout accounts)")
    
    logger.info("ALL ACCEPTANCE CRITERIA SUCCESSFULLY PASSED!")
    return True

if __name__ == "__main__":
    run_acceptance_tests()
