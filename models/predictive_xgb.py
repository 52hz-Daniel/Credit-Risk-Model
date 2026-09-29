import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_MASTER_PARQUET,
    SAVED_MODELS_DIR,
    RANDOM_SEED,
    CUSTOMER_ONLY_FEATURES,
    MACRO_FEATURES,
    TARGET_COL
)
from logger import logger
from models.feature_engineering import build_preprocessor

def calculate_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across probability deciles."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    # Weighted difference between predicted probability and actual frequency
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += prop * np.abs(bin_acc - bin_conf)
    return float(ece)

class DualCreditModelSuite:
    """
    Trains, calibrates, and evaluates both Customer-Only and Macro-Aware XGBoost models.
    Adheres to OSFI Guideline E-23 model risk governance and calibration standards.
    """
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.cat_cols = ["sex", "education", "marriage"]
        
        self.base_pipeline = None
        self.base_model = None
        self.macro_pipeline = None
        self.macro_model = None
        
        self.evaluation_results = {}
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepares train/test sets and separates feature spaces."""
        y = df[TARGET_COL].values
        
        # Customer-only features
        X_cust = df[CUSTOMER_ONLY_FEATURES].copy()
        
        # Macro-aware features
        all_features = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
        X_macro = df[all_features].copy()
        
        # 80/20 train/test split with stratification
        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=0.20,
            random_state=self.random_state,
            stratify=y
        )
        
        return X_cust, X_macro, y, idx_train, idx_test

    def train_and_calibrate(self, df: pd.DataFrame):
        X_cust, X_macro, y, idx_train, idx_test = self.prepare_data(df)
        
        y_train, y_test = y[idx_train], y[idx_test]
        scale_pos_weight = float((len(y_train) - np.sum(y_train)) / np.sum(y_train))
        
        logger.info(f"Training models with scale_pos_weight={scale_pos_weight:.2f} on {len(idx_train)} rows")
        
        # ----------------------------------------------------
        # 1. Customer-Only Baseline Model
        # ----------------------------------------------------
        logger.info("Fitting Customer-Only XGBoost Pipeline...")
        num_cust = [c for c in CUSTOMER_ONLY_FEATURES if c not in self.cat_cols]
        preproc_cust = build_preprocessor(num_cust, self.cat_cols)
        
        X_cust_train_trans = preproc_cust.fit_transform(X_cust.iloc[idx_train])
        X_cust_test_trans = preproc_cust.transform(X_cust.iloc[idx_test])
        
        xgb_base = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss"
        )
        # Out-of-fold Platt scaling probability calibration
        calib_base = CalibratedClassifierCV(estimator=xgb_base, method="sigmoid", cv=5)
        calib_base.fit(X_cust_train_trans, y_train)
        
        self.base_pipeline = preproc_cust
        self.base_model = calib_base
        
        # ----------------------------------------------------
        # 2. Macro-Aware Model
        # ----------------------------------------------------
        logger.info("Fitting Macro-Aware XGBoost Pipeline...")
        all_features = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
        num_macro = [c for c in all_features if c not in self.cat_cols]
        preproc_macro = build_preprocessor(num_macro, self.cat_cols)
        
        X_macro_train_trans = preproc_macro.fit_transform(X_macro.iloc[idx_train])
        X_macro_test_trans = preproc_macro.transform(X_macro.iloc[idx_test])
        
        xgb_macro = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss"
        )
        calib_macro = CalibratedClassifierCV(estimator=xgb_macro, method="sigmoid", cv=5)
        calib_macro.fit(X_macro_train_trans, y_train)
        
        self.macro_pipeline = preproc_macro
        self.macro_model = calib_macro
        
        # ----------------------------------------------------
        # 3. Comprehensive Evaluation
        # ----------------------------------------------------
        prob_base = calib_base.predict_proba(X_cust_test_trans)[:, 1]
        prob_macro = calib_macro.predict_proba(X_macro_test_trans)[:, 1]
        
        self.evaluation_results = {
            "customer_only": {
                "roc_auc": float(roc_auc_score(y_test, prob_base)),
                "pr_auc": float(average_precision_score(y_test, prob_base)),
                "brier_score": float(brier_score_loss(y_test, prob_base)),
                "ece": float(calculate_expected_calibration_error(y_test, prob_base)),
                "log_loss": float(log_loss(y_test, prob_base))
            },
            "macro_aware": {
                "roc_auc": float(roc_auc_score(y_test, prob_macro)),
                "pr_auc": float(average_precision_score(y_test, prob_macro)),
                "brier_score": float(brier_score_loss(y_test, prob_macro)),
                "ece": float(calculate_expected_calibration_error(y_test, prob_macro)),
                "log_loss": float(log_loss(y_test, prob_macro))
            }
        }
        
        # Persist models and test payload
        joblib.dump(self.base_pipeline, SAVED_MODELS_DIR / "cust_pipeline.joblib")
        joblib.dump(self.base_model, SAVED_MODELS_DIR / "cust_model.joblib")
        joblib.dump(self.macro_pipeline, SAVED_MODELS_DIR / "macro_pipeline.joblib")
        joblib.dump(self.macro_model, SAVED_MODELS_DIR / "macro_model.joblib")
        
        # Save test dataset and predictions for SHAP and dashboard
        test_payload = {
            "idx_test": idx_test,
            "y_test": y_test,
            "prob_base": prob_base,
            "prob_macro": prob_macro,
            "X_cust_test": X_cust.iloc[idx_test],
            "X_macro_test": X_macro.iloc[idx_test],
            "customer_ids": df["customer_id"].iloc[idx_test].values
        }
        joblib.dump(test_payload, SAVED_MODELS_DIR / "test_payload.joblib")
        joblib.dump(self.evaluation_results, SAVED_MODELS_DIR / "metrics.joblib")
        
        logger.info("Dual models trained, calibrated, evaluated, and saved to disk.")
        return self.evaluation_results

if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    suite = DualCreditModelSuite()
    metrics = suite.train_and_calibrate(df)
    
    print("\n" + "="*50)
    print("      HEAD-TO-HEAD MODEL EVALUATION REPORT")
    print("="*50)
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.to_string())
    print("="*50)
