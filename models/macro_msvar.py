import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REAL_MACRO_CSV
from logger import logger

class MacroRegimeEngine:
    """
    Markov-Switching Regime Engine based on Hamilton (1989, 1990).
    Fits a 2-state regime model on macroeconomic financial stress / spreads
    and calculates non-look-ahead filtered probabilities and systemic_risk_index.
    """
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        self.stress_regime_idx = 1
        
    def fit(self, macro_df: pd.DataFrame, target_col: str = "high_yield_spread") -> pd.DataFrame:
        """
        Fits a 2-state Markov Switching model on the macro target series.
        Applies deterministic labeling: Stress is the state with higher conditional mean.
        """
        logger.info(f"Fitting Hamilton Markov-Switching model on {target_col} (N={len(macro_df)} months)")
        y = macro_df[target_col].dropna()
        
        # Fit 2-state Markov Switching Regression with switching mean and stable variance (Hamilton 1990)
        try:
            self.model = MarkovRegression(
                endog=y,
                k_regimes=self.n_regimes,
                trend='c',
                switching_trend=True,
                switching_variance=False
            )
            self.results = self.model.fit(disp=False, maxiter=200, search_reps=20)
        except Exception as e:
            logger.warning(f"Default fit failed with {e}, falling back to MarkovAutoregression(order=1)")
            self.model = MarkovAutoregression(
                endog=y,
                k_regimes=self.n_regimes,
                order=1,
                switching_ar=False
            )
            self.results = self.model.fit(disp=False, maxiter=200)

        # Deterministic Stress Labeling: Regime with higher mean is the Stress regime
        params = self.results.params
        mean_keys = [k for k in params.index if "const" in k]
        if len(mean_keys) >= 2:
            means = [params[k] for k in mean_keys[:self.n_regimes]]
            self.stress_regime_idx = int(np.argmax(means))
        else:
            self.stress_regime_idx = 1
            
        logger.info(f"Identified Stress Regime as State {self.stress_regime_idx}")
        
        # Extract Filtered Marginal Probabilities (P(S_t | F_t)) to avoid future look-ahead bias
        filtered_probs = self.results.filtered_marginal_probabilities
        stress_prob_series = filtered_probs[self.stress_regime_idx]
        
        macro_df = macro_df.copy()
        macro_df["regime_prob_stress"] = stress_prob_series
        # Systemic Risk Index is clamped to [0, 1]
        macro_df["systemic_risk_index"] = np.clip(macro_df["regime_prob_stress"], 0.0, 1.0)
        
        # Save enriched macro series with Hamilton systemic risk index to CSV
        macro_df.to_csv(REAL_MACRO_CSV)
        logger.info(f"Updated {REAL_MACRO_CSV} with Hamilton systemic_risk_index")
        return macro_df

if __name__ == "__main__":
    from utils.macro_data import build_macro_dataset
    macro_data = build_macro_dataset(60)
    engine = MacroRegimeEngine()
    enriched_macro = engine.fit(macro_data)
    print("\nEnriched Macro with Hamilton Systemic Risk Index:")
    print(enriched_macro[["high_yield_spread", "regime_prob_stress", "systemic_risk_index"]].tail(10))
