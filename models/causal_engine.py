import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

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

class MultiArmXLearner:
    """
    Multi-Arm X-Learner for Heterogeneous Treatment Effect (CATE) Estimation.
    Reference: Künzel, Sekhon, Bickel, Yu (PNAS 2019) 'Metalearners for estimating
    heterogeneous treatment effects using machine learning'.
    
    Interventions:
      T = 0: Control (No Action / Baseline terms)
      T = 1: Limit Cut 20% (Immediate line reduction)
      T = 2: Payment Holiday (Temporary forbearance)
    
    Estimates:
      tau_{1,0}(x) = E[Y(1) - Y(0) | X = x]  (Risk difference of Limit Cut)
      tau_{2,0}(x) = E[Y(2) - Y(0) | X = x]  (Risk difference of Holiday)
    """
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.cat_cols = ["sex", "education", "marriage"]
        self.preprocessor = None
        
        # Stage 1: Base outcome models mu_t(x)
        self.base_models = {}
        # Stage 2: Propensity score models e_t(x)
        self.propensity_models = {}
        # Stage 3: Second-stage effect models tau_{t,0} and tau_{t,1}
        self.effect_models = {}
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray):
        """
        Fits the multi-arm X-learner using XGBoost base and effect models.
        """
        logger.info(f"Fitting Multi-Arm X-Learner on N={len(X)} samples across arms: {np.unique(T)}")
        
        # 1. Feature Preprocessing
        feature_cols = list(X.columns)
        num_cols = [c for c in feature_cols if c not in self.cat_cols]
        self.preprocessor = build_preprocessor(num_cols, self.cat_cols)
        X_trans = self.preprocessor.fit_transform(X)
        
        # 2. Stage 1: Estimate base outcome models mu_t(x) = E[Y | T=t, X=x]
        for arm in [0, 1, 2]:
            mask_arm = (T == arm)
            logger.info(f"Training Stage 1 base outcome model for Arm {arm} (N={np.sum(mask_arm)})")
            base_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
                eval_metric="logloss"
            )
            base_clf.fit(X_trans[mask_arm], y[mask_arm])
            self.base_models[arm] = base_clf
            
        # 3. Stages 2-4: Pairwise X-Learner for Treatment 1 and Treatment 2 vs Control (0)
        X_0 = X_trans[T == 0]
        y_0 = y[T == 0]
        
        for treat_arm in [1, 2]:
            logger.info(f"Computing X-Learner counterfactual effects for Arm {treat_arm} vs Control...")
            mask_t = (T == treat_arm)
            X_t = X_trans[mask_t]
            y_t = y[mask_t]
            
            # Step A: Propensity model e_t(x) = P(T = treat_arm | T in {0, treat_arm}, X)
            mask_pair = (T == 0) | (T == treat_arm)
            T_pair = (T[mask_pair] == treat_arm).astype(int)
            prop_clf = LogisticRegression(max_iter=500, random_state=self.random_state)
            prop_clf.fit(X_trans[mask_pair], T_pair)
            self.propensity_models[treat_arm] = prop_clf
            
            # Step B: Impute counterfactuals
            # For treated units: D_t = Y_t - mu_0(X_t)
            mu_0_pred_on_t = self.base_models[0].predict_proba(X_t)[:, 1]
            D_t = y_t - mu_0_pred_on_t
            
            # For control units: D_0 = mu_t(X_0) - Y_0
            mu_t_pred_on_0 = self.base_models[treat_arm].predict_proba(X_0)[:, 1]
            D_0 = mu_t_pred_on_0 - y_0
            
            # Step C: Train second-stage effect regressors
            tau_t = xgb.XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
            tau_t.fit(X_t, D_t)
            
            tau_0 = xgb.XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
            tau_0.fit(X_0, D_0)
            
            self.effect_models[treat_arm] = {"model_t": tau_t, "model_0": tau_0}
            
        logger.info("Multi-Arm X-Learner training complete.")
        return self

    def predict_cate(self, X: pd.DataFrame) -> dict:
        """
        Estimates Conditional Average Treatment Effects (CATE) tau_{a,0}(x)
        for Limit Cut (Arm 1) and Payment Holiday (Arm 2) relative to Control (0).
        """
        X_trans = self.preprocessor.transform(X)
        cate_estimates = {}
        
        for treat_arm in [1, 2]:
            prop_model = self.propensity_models[treat_arm]
            e_t = prop_model.predict_proba(X_trans)[:, 1]
            
            tau_t_pred = self.effect_models[treat_arm]["model_t"].predict(X_trans)
            tau_0_pred = self.effect_models[treat_arm]["model_0"].predict(X_trans)
            
            # Convex combination weighted by propensity: e(x)*tau_0 + (1 - e(x))*tau_t
            tau_combined = e_t * tau_0_pred + (1.0 - e_t) * tau_t_pred
            cate_estimates[f"tau_{treat_arm}_0"] = tau_combined
            
        return cate_estimates

    def prescribe_action(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prescribes optimal credit intervention policy pi*(x) to minimize 90-day default risk:
        pi*(x) = argmin_{t in {0,1,2}} E[Y(t) | X=x].
        """
        X_trans = self.preprocessor.transform(X)
        
        # Base risk under Control (T=0)
        p_control = self.base_models[0].predict_proba(X_trans)[:, 1]
        
        cate = self.predict_cate(X)
        tau_1 = cate["tau_1_0"] # Limit Cut risk delta
        tau_2 = cate["tau_2_0"] # Payment Holiday risk delta
        
        p_cut = np.clip(p_control + tau_1, 0.0, 1.0)
        p_holiday = np.clip(p_control + tau_2, 0.0, 1.0)
        
        out_df = pd.DataFrame({
            "pd_control": p_control,
            "pd_limit_cut": p_cut,
            "pd_payment_holiday": p_holiday,
            "tau_limit_cut": tau_1,
            "tau_payment_holiday": tau_2,
            "uplift_limit_cut": -tau_1,
            "uplift_payment_holiday": -tau_2
        })
        
        # Optimal intervention minimizing expected default probability
        risks = np.vstack([p_control, p_cut, p_holiday]).T
        actions = np.argmin(risks, axis=1)
        action_names = {0: "Do Nothing (Control)", 1: "Limit Cut 20%", 2: "Payment Holiday"}
        out_df["recommended_action"] = [action_names[a] for a in actions]
        out_df["action_code"] = actions
        
        return out_df

    def save(self, path):
        payload = {
            "preprocessor": self.preprocessor,
            "base_models": self.base_models,
            "propensity_models": self.propensity_models,
            "effect_models": self.effect_models,
            "random_state": self.random_state
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        inst = cls(random_state=payload["random_state"])
        inst.preprocessor = payload["preprocessor"]
        inst.base_models = payload["base_models"]
        inst.propensity_models = payload["propensity_models"]
        inst.effect_models = payload["effect_models"]
        return inst

def train_and_persist_causal_engine():
    """Trains and saves the X-Learner causal engine."""
    df = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    feature_cols = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
    
    X = df[feature_cols].copy()
    T = df["treatment_group"].values
    y = df[TARGET_COL].values
    
    engine = MultiArmXLearner()
    engine.fit(X, T, y)
    
    # Save to disk
    causal_path = SAVED_MODELS_DIR / "causal_xlearner.joblib"
    engine.save(causal_path)
    logger.info(f"Saved trained X-Learner causal engine payload to {causal_path}")
    
    # Validation evaluation sample
    sample_recs = engine.prescribe_action(X.head(10))
    print("\nSample Causal Prescriptions & Treatment Uplift:")
    print(sample_recs[["pd_control", "pd_limit_cut", "pd_payment_holiday", "recommended_action"]].head())
    
    return engine

if __name__ == "__main__":
    train_and_persist_causal_engine()
