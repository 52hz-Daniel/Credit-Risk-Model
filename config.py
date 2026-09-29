from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

DATA_DIR.mkdir(exist_ok=True)
SAVED_MODELS_DIR.mkdir(exist_ok=True)

# Random Seeds
RANDOM_SEED = 42

# Real Data Paths
REAL_CREDIT_PARQUET = DATA_DIR / "real_credit_card_clients.parquet"
REAL_MACRO_CSV = DATA_DIR / "real_macro_series.csv"
PROCESSED_MASTER_PARQUET = DATA_DIR / "master_credit_portfolio.parquet"

# Central Bank & Macro API Endpoints
BOC_VALET_BASE = "https://www.bankofcanada.ca/valet"
BOC_SERIES = {
    "policy_rate": "V39079",           # Bank of Canada Target Overnight Rate
    "cpi_inflation": "STATIC_OR_VALET" # Canadian Headline CPI or YoY
}

# FRED CSV series URLs (Public Direct Download)
FRED_SERIES_URLS = {
    "unemployment_rate": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
    "high_yield_spread": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2",
    "financial_stress": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4"
}

# Model Specifications
CUSTOMER_ONLY_FEATURES = [
    "limit_bal",
    "age",
    "sex",
    "education",
    "marriage",
    "pay_max_delinquency",
    "pay_recent_delinquency",
    "revolving_utilization",
    "avg_bill_amt",
    "avg_pay_amt",
    "pay_to_bill_ratio",
    "bill_to_limit_ratio"
]

MACRO_FEATURES = [
    "policy_rate",
    "high_yield_spread",
    "unemployment_rate",
    "systemic_risk_index"
]

TARGET_COL = "default_next_90_days"
