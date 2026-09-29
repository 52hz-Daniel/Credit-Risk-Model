import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAVED_MODELS_DIR, MACRO_FEATURES, CUSTOMER_ONLY_FEATURES
from logger import logger

class ModelExplainabilityEngine:
    """
    OSFI Guideline E-23 compliant Explainable AI module using Lundberg & Lee (2017) SHAP.
    Computes mathematically axiomatic Shapley attribution values for both
    portfolio-wide risk drivers and individual customer adverse action explanations.
    """
    def __init__(self, model_type: str = "macro_aware"):
        self.model_type = model_type
        if model_type == "macro_aware":
            self.pipeline = joblib.load(SAVED_MODELS_DIR / "macro_pipeline.joblib")
            self.calib_model = joblib.load(SAVED_MODELS_DIR / "macro_model.joblib")
            self.feature_names = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
        else:
            self.pipeline = joblib.load(SAVED_MODELS_DIR / "cust_pipeline.joblib")
            self.calib_model = joblib.load(SAVED_MODELS_DIR / "cust_model.joblib")
            self.feature_names = CUSTOMER_ONLY_FEATURES
            
        self.test_payload = joblib.load(SAVED_MODELS_DIR / "test_payload.joblib")
        # Extract underlying tree model from the calibrated classifier
        self.base_xgb = self.calib_model.calibrated_classifiers_[0].estimator
        self.explainer = shap.TreeExplainer(self.base_xgb)
        
        # Get transformed feature names
        self.transformed_feature_names = self._get_feature_names()
        
    def _get_feature_names(self):
        """Extracts readable feature names from the fitted ColumnTransformer."""
        num_cols = [c for c in self.feature_names if c not in ["sex", "education", "marriage"]]
        cat_encoder = self.pipeline.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(["sex", "education", "marriage"]).tolist()
        return num_cols + cat_names

    def get_global_shap_importance(self, sample_size: int = 500) -> pd.DataFrame:
        """Computes mean absolute SHAP importance values across test population."""
        X_test = self.test_payload["X_macro_test"] if self.model_type == "macro_aware" else self.test_payload["X_cust_test"]
        X_sample = X_test.iloc[:sample_size]
        X_trans = self.pipeline.transform(X_sample)
        
        shap_values = self.explainer.shap_values(X_trans)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        df_importance = pd.DataFrame({
            "feature": self.transformed_feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        
        return df_importance

    def generate_local_explanation(self, customer_idx: int = 0) -> dict:
        """
        Generates local Shapley contributions for a single customer decision.
        Formats the output for OSFI E-23 compliant adverse action reporting.
        """
        X_test = self.test_payload["X_macro_test"] if self.model_type == "macro_aware" else self.test_payload["X_cust_test"]
        row_raw = X_test.iloc[customer_idx:customer_idx+1]
        row_trans = self.pipeline.transform(row_raw)
        
        # Raw log-odds SHAP values
        shap_vals = self.explainer.shap_values(row_trans)[0]
        base_value = float(self.explainer.expected_value)
        
        # Predicted probability from the calibrated model
        prob = float(self.calib_model.predict_proba(row_trans)[0, 1])
        
        contributions = pd.DataFrame({
            "feature": self.transformed_feature_names,
            "shap_value": shap_vals,
            "abs_impact": np.abs(shap_vals)
        }).sort_values(by="abs_impact", ascending=False).reset_index(drop=True)
        
        return {
            "customer_id": int(self.test_payload["customer_ids"][customer_idx]),
            "predicted_pd": prob,
            "base_value": base_value,
            "contributions": contributions,
            "raw_features": row_raw.iloc[0].to_dict()
        }

if __name__ == "__main__":
    engine = ModelExplainabilityEngine("macro_aware")
    global_imp = engine.get_global_shap_importance(500)
    print("\nTop 10 Global SHAP Risk Drivers (Macro-Aware):")
    print(global_imp.head(10).to_string(index=False))
    
    local_exp = engine.generate_local_explanation(0)
    print(f"\nLocal OSFI E-23 Explanation for Customer #{local_exp['customer_id']}:")
    print(f"Predicted PD: {local_exp['predicted_pd']:.2%}")
    print(local_exp['contributions'].head(5).to_string(index=False))
