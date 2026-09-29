# AI Retail Credit Risk & Macroeconomic Regime Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-XGBoost%20%7C%20Statsmodels%20%7C%20SHAP-orange)](https://github.com/52hz-Daniel/Credit-Risk-Model)
[![Compliance: OSFI E-23](https://img.shields.io/badge/Compliance-OSFI%20E--23%20%7C%20SR%2011--7-green)](https://www.osfi-bsif.gc.ca/)

An institutional-grade **Retail Credit Risk Decisioning and Macroeconomic Regime Platform** designed to estimate 90-day borrower default probabilities under shifting economic regimes, optimize risk-adjusted margins, and provide mathematically auditable adverse-action reasons under **OSFI Guideline E-23** and **Basel III / SR 11-7** model governance standards.

---

## 🏛️ Executive Summary & Key Highlights

Traditional retail credit scorecards (e.g., standard FICO cut-offs or static logistic regressions) are backward-looking and slow to adapt during macroeconomic regime transitions (e.g., rate-hike cycles, commodity volatility, or credit spread widening).

This repository integrates:
1. **Hamilton (1989, 1990) Markov-Switching Autoregression (MS-AR)** on real **Bank of Canada Valet** and **FRED** macro time series to extract filtered, non-look-ahead systemic stress probabilities.
2. **Shannon Information Entropy Screening**: Ranking predictive risk features by Information Gain $IG(Y, X) = H(Y) - H(Y \mid X)$ without prior model bias.
3. **Dual Calibrated XGBoost Architecture**: Head-to-head comparison of a Customer-Only baseline against a Macro-Aware model equipped with out-of-fold probability calibration (Platt scaling / Isotonic regression).
4. **Explainable AI (XAI)**: Mathematically axiomatic **TreeSHAP** (Lundberg & Lee 2017) decomposing portfolio-wide risk drivers and producing compliant adverse-action waterfall explanations for individual loan applicants.
5. **Interactive Executive Dashboard**: A full-featured Streamlit application featuring live macro stress-testing sliders, ROC / PR curves, and reliability calibration diagrams.

---

## 📊 Empirical Model Performance (Holdout Evaluation)

Evaluated on a 6,000-borrower holdout test set from an empirical 30,000-account credit portfolio:

| Evaluation Metric | Customer-Only Baseline | Macro-Aware Model | Benchmark Standard |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.7800** | **0.7792** | High discriminative power separating defaulters from non-defaulters |
| **PR-AUC (Avg. Precision)** | **0.5588** | **0.5574** | Strong positive predictive value under class imbalance (22% default rate) |
| **Brier Score** | **0.1355** | **0.1356** | Low mean-squared probability error (optimal $\to 0$) |
| **Expected Calibration Error (ECE)** | **0.0173** | **0.0170** | **1.7% decile deviation** (exceptional probability calibration) |
| **Log Loss** | **0.4310** | **0.4314** | Tight cross-entropy fit |

---

## 📐 Mathematical Architecture

### 1. Macro Regime Detection (Hamilton 1989, 1990)
Macroeconomic conditions shift between discrete hidden states $S_t \in \{0, 1\}$ (Calm vs. Stress):
$$y_t = \mu_{S_t} + \sum_{j=1}^p \phi_j (y_{t-j} - \mu_{S_{t-j}}) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$
* **Filtered Marginal Probability**: Filtered probabilities $\Pr(S_t = \text{Stress} \mid \mathcal{F}_t)$ are computed forward without future look-ahead bias.
* **Systemic Risk Index**: Monotonically clamped into $[0, 1]$ to serve as a stable macroeconomic covariate for retail scoring.

### 2. Shannon Entropy & Information Gain
Features are screened by measuring the reduction in informational uncertainty:
$$H(Y) = -\sum_{y \in \{0, 1\}} p(y) \log_2 p(y)$$
$$IG(Y, X) = H(Y) - \sum_{x} p(x) H(Y \mid X = x)$$
Empirically, repayment status indicators (`pay_recent_delinquency` and `pay_max_delinquency`) lead with $>0.10$ bits of Information Gain.

### 3. Out-of-Fold Probability Calibration
Raw tree log-odds are mapped to well-calibrated true posterior probabilities via Platt sigmoid scaling:
$$\hat{p}_i = \frac{1}{1 + \exp(A \cdot f(x_i) + B)}$$
Fitted strictly using 5-fold cross-validation to prevent calibration leakage.

### 4. OSFI Guideline E-23 SHAP Explainability
Individual credit decisions are decomposed into additive Shapley attributions:
$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
Satisfying Efficiency, Symmetry, and Additivity axioms, allowing automated generation of adverse action notices.

---

## 📁 Repository Structure

```
├── app/
│   └── main_dashboard.py          # Interactive Streamlit Comparative Dashboard
├── docs/
│   ├── ms_regime_foundations.md   # Hamilton MS-AR specification & diagnostics
│   ├── predictive_model_spec.md   # Dual XGBoost and probability calibration spec
│   └── causal_spec.md             # Multi-arm CATE intervention engine spec
├── models/
│   ├── macro_msvar.py             # Hamilton Markov-Switching regime engine
│   ├── feature_engineering.py     # Shannon Entropy & Information Gain ranking
│   ├── predictive_xgb.py          # Dual XGBoost training & probability calibration
│   └── explainability.py          # SHAP TreeExplainer & local waterfall module
├── utils/
│   └── macro_data.py              # Ingestion from Bank of Canada Valet & FRED APIs
├── tests/
│   └── verify_pipeline.py         # Automated quantitative validation & acceptance test suite
├── config.py                      # Centralized configuration & feature parameters
├── data_merger.py                 # Leakage-safe temporal as-of dataset merge
├── data_pipeline_customers.py     # Retail customer portfolio processing & risk ratios
├── logger.py                      # Traceable logging utility
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/52hz-Daniel/Credit-Risk-Model.git
cd Credit-Risk-Model
```

### 2. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note for macOS Apple Silicon users: Ensure OpenMP runtime is available: `brew install libomp`)*

### 3. Run Pipeline & Train Models
```bash
# Ingest macro series & compute Hamilton systemic risk index
python3 models/macro_msvar.py

# Build master dataset and train dual calibrated XGBoost models
python3 data_merger.py
python3 models/predictive_xgb.py

# Run acceptance tests
python3 tests/verify_pipeline.py
```

### 4. Launch the Executive Dashboard
```bash
streamlit run app/main_dashboard.py
```
Open your browser at `http://localhost:8501` to view:
* **Model Evaluation**: ROC, PR, and Reliability diagrams.
* **Macro Regimes & Stress Testing**: Real-time stress shock simulation sliders.
* **Feature Information Gain**: Shannon entropy feature rankings.
* **OSFI E-23 Explainability**: Interactive SHAP waterfall charts for any audited customer.

---

## 📜 Regulatory References & Governance
* **OSFI Guideline E-23**: Enterprise-Wide Model Risk Management for Deposit-Taking Institutions.
* **Federal Reserve SR 11-7 / OCC 2011-12**: Guidance on Model Risk Management.
* **Hamilton, J.D. (1989)**: *"A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle."* Econometrica.
* **Lundberg, S.M. & Lee, S.-I. (2017)**: *"A Unified Approach to Interpreting Model Predictions."* NeurIPS.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
