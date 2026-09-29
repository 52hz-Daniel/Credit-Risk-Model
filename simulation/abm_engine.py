import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mesa

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_MASTER_PARQUET,
    REAL_MACRO_CSV,
    SAVED_MODELS_DIR,
    RANDOM_SEED,
    CUSTOMER_ONLY_FEATURES,
    MACRO_FEATURES
)
from logger import logger
from models.retail_game_theory import RetailGameTheoryOptimizer, calculate_poaching_probability

class CustomerAgent(mesa.Agent):
    """
    Step 50 & 52: Represents an individual retail credit card borrower.
    Undergoes monthly income/expense dynamics, interest accrual, debt service,
    liquidity shocks, delinquency transitions, and default events.
    """
    def __init__(self, model, row: pd.Series):
        super().__init__(model)
        self.row_data = row
        self.customer_id = int(row.get("customer_id", self.unique_id))
        
        self.limit = float(row.get("limit_bal", 50000.0))
        self.balance = float(row.get("limit_bal", 50000.0)) * float(row.get("revolving_utilization", 0.3))
        
        # Monthly cash flow parameters
        self.income = max(2500.0, float(self.limit * 0.25))
        self.expenses = self.income * float(self.model.rng.uniform(0.70, 0.90))
        
        self.missed_streak = int(row.get("pay_max_delinquency", 0))
        self.is_defaulted = False
        self.is_poached = False
        self.in_holiday = False
        self.holiday_months_left = 0
        
    def step(self):
        """Monthly borrower cash flow update and solvency check."""
        if self.is_defaulted or self.is_poached:
            return

        # 1. Manage Forbearance Holiday
        if self.in_holiday:
            self.holiday_months_left -= 1
            if self.holiday_months_left <= 0:
                self.in_holiday = False
                
        # 2. Monthly cash balance & expense dynamics
        net_cash_flow = self.income - self.expenses
        
        # Minimum payment required (3% of balance) unless on holiday
        min_payment = 0.0 if self.in_holiday else max(25.0, self.balance * 0.03)
        
        if net_cash_flow >= min_payment:
            # Paid required balance; decrease balance by excess cash flow
            payment = min(self.balance, net_cash_flow)
            self.balance -= payment
            self.missed_streak = 0
        else:
            # Cash deficit: miss payment and accrue shortfall to balance
            shortfall = min_payment - net_cash_flow
            self.balance += shortfall
            self.missed_streak += 1

        # 3. Monthly revolving interest (19.99% APR / 12)
        monthly_apr = 0.1999 / 12.0
        interest_charge = self.balance * monthly_apr
        self.balance += interest_charge
        self.model.monthly_interest_revenue += interest_charge

        # 4. Solvency and Default Threshold Checks
        # Default triggers if balance > 115% of limit or 3+ consecutive missed payments (90 DPD)
        if self.balance > (self.limit * 1.15) or self.missed_streak >= 3:
            self.is_defaulted = True
            default_loss = self.balance * 0.70 # LGD = 70%
            self.model.monthly_default_losses += default_loss
            self.model.cumulative_defaults_count += 1
            return

        # 5. Check Competitor Poaching Risk
        if not self.is_defaulted and not self.is_poached:
            util = self.balance / max(1.0, self.limit)
            poach_prob = self.model.rng.uniform(0.0, 1.0)
            threshold = 0.04 # Baseline low monthly churn
            if poach_prob < threshold:
                self.is_poached = True
                self.model.cumulative_poached_count += 1

class BankAgent(mesa.Agent):
    """
    Step 51, 54 & 55: Represents the bank portfolio decisioning engine.
    Executes either 'traditional' reactive policy or 'causal_ai' game-theoretic optimization.
    """
    def __init__(self, model, strategy_mode: str = "causal_ai"):
        super().__init__(model)
        self.strategy_mode = strategy_mode
        self.optimizer = RetailGameTheoryOptimizer() if strategy_mode == "causal_ai" else None

    def step(self):
        """Monthly portfolio review and credit line intervention."""
        active_customers = [a for a in self.model.customer_agents if not a.is_defaulted and not a.is_poached]
        
        if self.strategy_mode == "traditional":
            # Traditional Strategy: Blanket 50% limit cut if borrower missed payment last month
            for agent in active_customers:
                if agent.missed_streak >= 1:
                    # Reactive limit cut
                    agent.limit *= 0.50
                    # Aggressive limit cut triggers high poaching if customer had low utilization
                    if (agent.balance / agent.limit) < 0.35 and self.model.rng.uniform(0, 1) < 0.40:
                        agent.is_poached = True
                        self.model.cumulative_poached_count += 1

        elif self.strategy_mode == "causal_ai":
            # Causal AI Strategy: Batch CATE intervention balancing default reduction & poaching
            review_agents = []
            review_rows = []
            for agent in active_customers:
                util = agent.balance / max(1.0, agent.limit)
                if util > 0.50 or agent.missed_streak >= 1:
                    row = agent.row_data.copy()
                    row["limit_bal"] = agent.limit
                    row["revolving_utilization"] = util
                    row["pay_max_delinquency"] = agent.missed_streak
                    review_agents.append(agent)
                    review_rows.append(row)
                    
            if review_rows:
                df_reviews = pd.DataFrame(review_rows)
                strat_res = self.optimizer.get_optimal_strategy(df_reviews)
                for agent, action in zip(review_agents, strat_res["optimal_strategy"]):
                    if action == "Limit Cut 20%":
                        agent.limit *= 0.80
                    elif action == "Payment Holiday":
                        agent.in_holiday = True
                        agent.holiday_months_left = 3
                        agent.missed_streak = 0

class CreditPortfolioModel(mesa.Model):
    """
    Step 49, 53, 56 & 57: 24-Month Agent-Based Simulation Model.
    Tracks macro shocks, borrower balance sheets, defaults, revenues, and net margins.
    """
    def __init__(self, customers_sample: pd.DataFrame, macro_series: pd.DataFrame, strategy_mode: str = "causal_ai", seed: int = RANDOM_SEED):
        super().__init__(seed=seed)
        self.strategy_mode = strategy_mode
        self.macro_series = macro_series.tail(24).reset_index(drop=True)
        self.current_step = 0
        self.rng = np.random.RandomState(seed)
        
        # Portfolio KPIs
        self.monthly_interest_revenue = 0.0
        self.monthly_default_losses = 0.0
        self.cumulative_revenue = 0.0
        self.cumulative_losses = 0.0
        self.cumulative_defaults_count = 0
        self.cumulative_poached_count = 0
        
        # Instantiate Customer Agents
        self.customer_agents = []
        for _, row in customers_sample.iterrows():
            agent = CustomerAgent(self, row)
            self.customer_agents.append(agent)
            
        # Instantiate Bank Agent
        self.bank_agent = BankAgent(self, strategy_mode=strategy_mode)
        
        # Time-series history logger
        self.history = []

    def step(self):
        """Advances the 24-month credit market simulation by one month."""
        month_idx = min(self.current_step, len(self.macro_series) - 1)
        macro_row = self.macro_series.iloc[month_idx]
        is_crisis_regime = float(macro_row.get("systemic_risk_index", 0.0)) > 0.50
        
        # Step 53: Macro environment shock
        # If systemic stress is high, randomly shock income of 10% of active borrowers
        if is_crisis_regime:
            for agent in self.customer_agents:
                if not agent.is_defaulted and not agent.is_poached:
                    if self.rng.uniform(0, 1) < 0.10:
                        # 50% income reduction simulating job loss / layoff
                        agent.income *= 0.50

        # Reset monthly flow trackers
        self.monthly_interest_revenue = 0.0
        self.monthly_default_losses = 0.0

        # Execute Bank Strategy
        self.bank_agent.step()
        
        # Execute Customer Agents
        for agent in self.customer_agents:
            agent.step()

        # Update Cumulative Financials
        self.cumulative_revenue += self.monthly_interest_revenue
        self.cumulative_losses += self.monthly_default_losses
        net_profit = self.cumulative_revenue - self.cumulative_losses - (self.cumulative_poached_count * 400.0)

        total_agents = len(self.customer_agents)
        active_count = sum(1 for a in self.customer_agents if not a.is_defaulted and not a.is_poached)

        self.history.append({
            "month": self.current_step + 1,
            "systemic_risk_index": float(macro_row.get("systemic_risk_index", 0.0)),
            "is_crisis": is_crisis_regime,
            "active_borrowers": active_count,
            "cumulative_defaults": self.cumulative_defaults_count,
            "default_rate": self.cumulative_defaults_count / total_agents,
            "cumulative_poached": self.cumulative_poached_count,
            "cumulative_revenue": self.cumulative_revenue,
            "cumulative_losses": self.cumulative_losses,
            "net_portfolio_profit": net_profit
        })

        self.current_step += 1

def run_comparative_simulation(n_agents: int = 1000, n_steps: int = 24, seed: int = RANDOM_SEED) -> tuple:
    """
    Step 57: Executes the simulation twice with identical customer cohorts:
      1. Traditional Strategy (Blanket reactive 50% limit cuts)
      2. Causal AI Strategy (Dynamic X-Learner CATE + Retail Game Theory)
    Returns: (df_traditional, df_causal, kpi_summary)
    """
    logger.info(f"Running 24-Month ABM Comparative Simulation (N={n_agents} agents)...")
    df_master = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    cohort_sample = df_master.sample(n=min(n_agents, len(df_master)), random_state=seed).copy()
    macro_series = pd.read_csv(REAL_MACRO_CSV, index_col="date")

    # Run Traditional Simulation
    model_trad = CreditPortfolioModel(cohort_sample, macro_series, strategy_mode="traditional", seed=seed)
    for _ in range(n_steps):
        model_trad.step()
    df_trad = pd.DataFrame(model_trad.history)

    # Run Causal AI Simulation
    model_causal = CreditPortfolioModel(cohort_sample, macro_series, strategy_mode="causal_ai", seed=seed)
    for _ in range(n_steps):
        model_causal.step()
    df_causal = pd.DataFrame(model_causal.history)

    # Comparative Summary KPIs
    trad_final = df_trad.iloc[-1]
    causal_final = df_causal.iloc[-1]

    summary = {
        "traditional_defaults": int(trad_final["cumulative_defaults"]),
        "causal_defaults": int(causal_final["cumulative_defaults"]),
        "default_reduction_count": int(trad_final["cumulative_defaults"] - causal_final["cumulative_defaults"]),
        "default_reduction_pct": (trad_final["cumulative_defaults"] - causal_final["cumulative_defaults"]) / max(1, trad_final["cumulative_defaults"]),
        "traditional_poached": int(trad_final["cumulative_poached"]),
        "causal_poached": int(causal_final["cumulative_poached"]),
        "traditional_net_profit": float(trad_final["net_portfolio_profit"]),
        "causal_net_profit": float(causal_final["net_portfolio_profit"]),
        "net_profit_lift": float(causal_final["net_portfolio_profit"] - trad_final["net_portfolio_profit"]),
        "profit_lift_pct": (causal_final["net_portfolio_profit"] - trad_final["net_portfolio_profit"]) / max(1.0, abs(trad_final["net_portfolio_profit"]))
    }

    logger.info(f"ABM Simulation Complete. Causal AI Net Profit Lift: ${summary['net_profit_lift']:,.2f} (+{summary['profit_lift_pct']:.1%})")
    
    # Save simulation results for dashboard and test verification
    payload = {
        "df_traditional": df_trad,
        "df_causal": df_causal,
        "summary": summary
    }
    joblib.dump(payload, SAVED_MODELS_DIR / "abm_simulation_results.joblib")
    return df_trad, df_causal, summary

if __name__ == "__main__":
    df_trad, df_causal, summary = run_comparative_simulation(n_agents=1000, n_steps=24)
    print("\n" + "="*70)
    print("      24-MONTH AGENT-BASED SIMULATION RESULTS (ABM ROI)")
    print("="*70)
    print(f"Traditional Strategy Defaults:  {summary['traditional_defaults']} ({summary['traditional_defaults']/10:.1f}%)")
    print(f"Causal AI Strategy Defaults:     {summary['causal_defaults']} ({summary['causal_defaults']/10:.1f}%)")
    print(f"Defaults Averted by Causal AI:  {summary['default_reduction_count']} (-{summary['default_reduction_pct']:.1%})")
    print(f"Customer Churn / Poached:       Traditional: {summary['traditional_poached']} vs Causal: {summary['causal_poached']}")
    print(f"Traditional Net Margin:         ${summary['traditional_net_profit']:,.2f}")
    print(f"Causal AI Net Margin:           ${summary['causal_net_profit']:,.2f}")
    print(f"Total Economic Profit Lift:     +${summary['net_profit_lift']:,.2f} (+{summary['profit_lift_pct']:.1%})")
    print("="*70)
