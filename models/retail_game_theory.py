import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAVED_MODELS_DIR, PROCESSED_MASTER_PARQUET, CUSTOMER_ONLY_FEATURES, MACRO_FEATURES
from logger import logger
from models.causal_engine import MultiArmXLearner

def calculate_poaching_probability(
    row: pd.Series,
    treatment: int,
    base_attrition: float = 0.05
) -> float:
    """
    Step 44-45: Models competitor poaching risk in retail banking.
    If the bank aggressively cuts limits (T=1) on a prime, low-utilization,
    never-delinquent customer, competitor banks (e.g. Amex, RBC, TD) have a
    high probability of successfully acquiring / poaching them.
    
    Treatments:
      0: Control (No Action)
      1: Limit Cut 20%
      2: Payment Holiday
    """
    util = float(row.get("revolving_utilization", 0.5))
    max_delinq = float(row.get("pay_max_delinquency", 0))
    limit = float(row.get("limit_bal", 100000))
    
    if treatment == 1:
        # Aggressive limit cut on prime customers triggers high flight risk
        if util < 0.30 and max_delinq == 0:
            # High-limit, prime borrowers are actively headhunted
            if limit >= 200000:
                return 0.70
            return 0.55
        elif util < 0.50 and max_delinq <= 1:
            return 0.30
        else:
            # Risky / high utilization customers are unlikely to be poached by competitors
            return 0.08
    elif treatment == 2:
        # Payment holiday: customer appreciates forbearance; flight risk is low
        return 0.06
    else:
        # Control (Status quo): baseline industry credit card churn
        return base_attrition

class RetailGameTheoryOptimizer:
    """
    Unifies Causal Machine Learning (X-Learner CATE) with Retail Competitor Game Theory
    to prescribe optimal credit actions maximizing risk-adjusted net economic margin.
    """
    def __init__(self, causal_engine: MultiArmXLearner = None):
        if causal_engine is None:
            causal_path = SAVED_MODELS_DIR / "causal_xlearner.joblib"
            self.causal_engine = MultiArmXLearner.load(causal_path)
        else:
            self.causal_engine = causal_engine
            
        self.lgd = 0.70          # Loss Given Default (unsecured credit card standard)
        self.apr = 0.1999        # Standard purchase interest APR (19.99%)
        self.funding_cost = 0.04 # Bank cost of funds (4%)
        self.net_interest_margin = self.apr - self.funding_cost # ~15.99%

    def compute_customer_ltv(self, row: pd.Series) -> float:
        """Estimates customer Lifetime Value (LTV) to quantify poaching penalty."""
        limit = float(row.get("limit_bal", 100000))
        delinq = float(row.get("pay_max_delinquency", 0))
        util = float(row.get("revolving_utilization", 0.3))
        
        # Prime customers have higher lifetime value to competitors
        annual_revenue = limit * max(util, 0.15) * self.net_interest_margin
        # Discounted over 3-year expected retention horizon
        discount_factor = 2.4
        delinq_penalty = max(0.2, 1.0 - 0.25 * delinq)
        ltv = annual_revenue * discount_factor * delinq_penalty
        return max(300.0, float(ltv))

    def get_optimal_strategy(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 46-48: Evaluates CATE from the Causal Engine and computes
        the expected economic net profit for each treatment:
        
        E[Profit(t)] = Expected Net Margin(t) - Expected Default Loss(t) - Poaching Penalty(t)
        
        Where:
          - Expected Default Loss(t) = PD(t) * EAD(t) * LGD
          - Poaching Penalty(t) = P_poach(t) * LTV
        
        Returns the action maximizing risk-adjusted profit while minimizing flight risk.
        """
        # 1. Causal Engine potential outcome predictions
        prescriptions = self.causal_engine.prescribe_action(customer_df)
        
        results = []
        for idx in range(len(customer_df)):
            row = customer_df.iloc[idx]
            p_ctrl = prescriptions.iloc[idx]["pd_control"]
            p_cut = prescriptions.iloc[idx]["pd_limit_cut"]
            p_hol = prescriptions.iloc[idx]["pd_payment_holiday"]
            
            limit = float(row.get("limit_bal", 100000))
            util = float(row.get("revolving_utilization", 0.3))
            current_balance = limit * util
            ltv = self.compute_customer_ltv(row)
            
            # --- Treatment 0: Control ---
            ead_0 = current_balance
            rev_0 = ead_0 * self.net_interest_margin
            loss_0 = p_ctrl * ead_0 * self.lgd
            poach_0 = calculate_poaching_probability(row, 0)
            penalty_0 = poach_0 * ltv
            profit_0 = rev_0 - loss_0 - penalty_0
            
            # --- Treatment 1: Limit Cut 20% ---
            # Balance capped if limit cut below current balance
            ead_1 = min(current_balance, limit * 0.80)
            rev_1 = ead_1 * self.net_interest_margin
            loss_1 = p_cut * ead_1 * self.lgd
            poach_1 = calculate_poaching_probability(row, 1)
            penalty_1 = poach_1 * ltv
            profit_1 = rev_1 - loss_1 - penalty_1
            
            # --- Treatment 2: Payment Holiday ---
            ead_2 = current_balance * 1.02 # Temporary accrued interest
            rev_2 = ead_2 * (self.net_interest_margin * 0.85) # Forbearance concession
            loss_2 = p_hol * ead_2 * self.lgd
            poach_2 = calculate_poaching_probability(row, 2)
            penalty_2 = poach_2 * ltv
            profit_2 = rev_2 - loss_2 - penalty_2
            
            # Choose treatment maximizing expected economic profit
            profits = [profit_0, profit_1, profit_2]
            best_action_idx = int(np.argmax(profits))
            action_names = {0: "Do Nothing (Control)", 1: "Limit Cut 20%", 2: "Payment Holiday"}
            
            results.append({
                "pd_control": p_ctrl,
                "pd_limit_cut": p_cut,
                "pd_payment_holiday": p_hol,
                "poach_prob_control": poach_0,
                "poach_prob_limit_cut": poach_1,
                "poach_prob_payment_holiday": poach_2,
                "expected_profit_control": profit_0,
                "expected_profit_limit_cut": profit_1,
                "expected_profit_payment_holiday": profit_2,
                "optimal_strategy": action_names[best_action_idx],
                "optimal_profit": profits[best_action_idx]
            })
            
        return pd.DataFrame(results, index=customer_df.index)

if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    feature_cols = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
    sample_customers = df.head(10)
    
    optimizer = RetailGameTheoryOptimizer()
    strat_df = optimizer.get_optimal_strategy(sample_customers[feature_cols])
    
    print("\n" + "="*80)
    print("      RETAIL GAME THEORY OPTIMAL STRATEGY MATRIX (SAMPLE 5)")
    print("="*80)
    for i in range(5):
        s = strat_df.iloc[i]
        c = sample_customers.iloc[i]
        print(f"Customer #{c['customer_id']} (Limit: ${c['limit_bal']:,.0f}, Util: {c['revolving_utilization']:.1%}, MaxDelinq: {c['pay_max_delinquency']}):")
        print(f"  Control:      PD={s['pd_control']:.1%}, PoachRisk={s['poach_prob_control']:.1%}, NetProfit=${s['expected_profit_control']:,.2f}")
        print(f"  Limit Cut:    PD={s['pd_limit_cut']:.1%}, PoachRisk={s['poach_prob_limit_cut']:.1%}, NetProfit=${s['expected_profit_limit_cut']:,.2f}")
        print(f"  Holiday:      PD={s['pd_payment_holiday']:.1%}, PoachRisk={s['poach_prob_payment_holiday']:.1%}, NetProfit=${s['expected_profit_payment_holiday']:,.2f}")
        print(f"  >>> OPTIMAL ACTION: {s['optimal_strategy']} (Net Exp Profit: ${s['optimal_profit']:,.2f})\n")
    print("="*80)
