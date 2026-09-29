# Predictive Risk Model Spec (Step 1): Entropy/IG + XGBoost

## 0) Scope and non-goals (Step-1)
- **Goal**: Define a leakage-safe, calibrated *predictive* baseline model that outputs \(\hat p_{i,t} = P(\texttt{default_next_90_days}_{i,t}=1\mid X_{i,t})\) for each customer \(i\) at as-of time \(t\), using customer features + macro features + `systemic_risk_index`.
- **Non-goals (this step)**: causal identification and treatment policy optimization; SHAP governance and model risk management; ABM integration; productionization/serving.
- **Model class (baseline)**: gradient-boosted decision trees (XGBoost-style boosting) for binary classification with probabilistic output.

## 1) Target definition: `default_next_90_days` + data-generating assumptions

### 1.1 Target definition
Let \(t\) denote an *as-of* observation timestamp for a customer (the time when features are measured and a score would be produced).

Define the binary outcome:
\[
Y_{i,t} \equiv \texttt{default_next_90_days}_{i,t} = \mathbb{1}\{\text{default event for customer } i \text{ occurs in } (t, t+90\text{d}]\}.
\]

Default event definition is environment-dependent:
- **Synthetic (now)**: a simulated default event (e.g., delinquency state transition, missed-payment threshold, or Bernoulli draw) parameterized only by customer and macro state *available at time \(t\)*.
- **Real data (later)**: an institutionally defined default trigger (e.g., \(\ge 90\) DPD, charge-off, bankruptcy) mapped consistently into \(Y_{i,t}\).

### 1.2 Data-generating assumptions (modeling, not causal)
- **Prediction target**: conditional risk \(P(Y_{i,t}=1\mid X_{i,t})\). No causal interpretation is implied.
- **Weak stationarity (pragmatic)**: the mapping \(X\mapsto Y\) may drift over time; therefore **temporal validation and stability tests are mandatory**.
- **Label correctness**: labels must be computed using only events in \((t, t+90\text{d}]\); no feature may reference outcomes after \(t\).

## 2) Feature schema (customer + macro + `systemic_risk_index`)
All features must have a schema: **type**, **units**, **allowable range**, **missingness policy**, and **time index**.

### 2.1 Customer-level features (examples; extensible)
All customer features are measured at as-of time \(t\) and are either static (time-invariant) or snapshot values.

- **Identifiers**
  - `customer_id` (string/int): identifier only (must not be used as a predictive feature).

- **Demographics / tenure**
  - `age_years` (float, years)
  - `region` (categorical; values depend on synthetic generator / institution)
  - `is_newcomer` (binary in \(\{0,1\}\))
  - `months_on_book` (int, months, \(\ge 0\))

- **Credit profile / usage**
  - `income` (float, currency per year; if synthetic uses it)
  - `credit_limit` (float, currency)
  - `current_balance` (float, currency)
  - `utilization_ratio` (float, unitless; expected in \([0,1]\), may exceed \(1\) if over-limit is allowed)
  - `historical_missed_payments_12m` (int, count, \(\ge 0\))
  - `foreign_credit_score_equivalent` (float/int; present only when `is_newcomer=1`; otherwise missing by design)

**Acceptance requirements (schema)**
- Every feature used for modeling has: **type**, **units**, **allowed range**, **missingness policy**, and **time index**.
- The feature matrix excludes identifiers and direct label proxies by construction.

### 2.2 Macro features
Macro features are indexed by a macro time axis \(\tau\) (e.g., monthly). Examples include unemployment, inflation/CPI, overnight rate, supply chain pressure indices, sentiment indices.

- `macro_time_index` (monthly, e.g., \(\tau = 1,\dots,60\) for synthetic)
- Example macro variables (all with units/frequency documented by the macro data owner):
  - `cpi_inflation_yoy` (%, monthly)
  - `overnight_rate` (%, daily aggregated to monthly; ffill allowed only under explicit rules)
  - `unemployment_rate` (%, monthly)
  - `rss_sentiment_score_weekly_roll` (unitless in \([-1,1]\), weekly aggregated to monthly)
  - `social_stress_index` (unitless, weekly aggregated to monthly)

### 2.3 `systemic_risk_index` (units + timing)
`systemic_risk_index` is a macro-derived scalar at macro time \(\tau\).

- **Unit**: unitless risk score.
- **Expected range (placeholder)**: \([0, 1.5]\) based on “regime probability \(\times\) shock multiplier” concept in the master plan.
- **Timing**: computed at macro time \(\tau\) using only macro information available up to \(\tau\) (no look-ahead).

**Important**: the precise mathematical definition and final range are owned by Agent A; this spec treats it as a required macro feature with strict leakage constraints and a contract (below).

## 3) Temporal logic & leakage prevention (macro-to-customer join)
Leakage prevention is the primary acceptance gate.

### 3.1 Time indices
- Customer observation time: \(t\) (date/time).
- Macro index: \(\tau\) (e.g., month-end dates or month integers).

Define an availability mapping \(\tau(t)\) = the latest macro timestamp \(\tau\) such that macro data are **published/available** by time \(t\).

### 3.2 As-of join rule (must)
For each customer observation \((i,t)\), macro features must be joined using:
\[
X^{macro}_{i,t} = X^{macro}_{\tau(t)}
\]
where \(\tau(t)\) is the **most recent available** macro row at or before \(t\). This is an **as-of join** (last observation carried forward under availability constraints).

**Forbidden (examples)**
- Joining macro values at \(\tau > \tau(t)\).
- Centered rolling windows that incorporate future macro values.
- Global preprocessing statistics fit on all periods (e.g., scaling fit on full dataset).
- Any feature derived directly or indirectly from the label window \((t, t+90\text{d}]\).

### 3.3 Label window vs feature window
- Features are computed using data \(\le t\).
- Labels use outcomes in \((t, t+90\text{d}]\).
- Any feature derived from customer behavior within \((t, t+90\text{d}]\) is leakage.

### 3.4 Validation splits (temporal)
Primary validation must reflect deployment:
- **Preferred**: time-based split by as-of time \(t\) (train earlier, test later).
- **Secondary**: random split allowed only for early synthetic smoke-testing, and must be labeled “non-deployment-realistic.”

## 4) Entropy / information gain: explicit definitions + binning rules
Compute univariate information measures of each feature \(X_j\) with respect to \(Y\) to support feature screening and diagnostics (not as a substitute for model-based importance).

### 4.1 Binary target entropy
Let \(Y\in\{0,1\}\) with \(p \equiv P(Y=1)\). Shannon entropy (base-2):
\[
H(Y) = -\sum_{y\in\{0,1\}} P(Y=y)\log_2 P(Y=y)
= -p\log_2 p - (1-p)\log_2(1-p).
\]
Use the convention \(0\log 0 = 0\). In finite samples, probabilities are empirical frequencies.

### 4.2 Conditional entropy given a discrete/binned feature
For discrete \(X\) taking values \(x\in\mathcal{X}\):
\[
H(Y\mid X) = \sum_{x\in\mathcal{X}} P(X=x)\, H(Y\mid X=x)
\]
with
\[
H(Y\mid X=x) = -\sum_{y\in\{0,1\}} P(Y=y\mid X=x)\log_2 P(Y=y\mid X=x).
\]

### 4.3 Information gain / mutual information
Define **information gain** as mutual information:
\[
IG(Y;X) \equiv I(Y;X) = H(Y) - H(Y\mid X) \ge 0.
\]

### 4.4 Binning rules (numerical features)
To compute \(IG(Y;X_j)\) when \(X_j\) is continuous/numeric, discretize \(X_j\to \tilde X_j\) using one of the following rules (must be chosen and documented per feature type):

- **Default**: quantile bins (equal-frequency)
  - number of bins \(B = 10\) (deciles) for \(N\ge 5000\), else \(B=5\).
  - ties handled by merging adjacent bins until all bins have \(\ge 1\%\) of samples.
- **Monotone ratio features** (e.g., utilization): domain-informed bins (example)
  - \([0,0.1),[0.1,0.3),[0.3,0.5),[0.5,0.8),[0.8,1.0),[1.0,\infty)\).
- **Count features**: cap at a max bucket and treat as discrete (example)
  - \(0,1,2,3,\ge4\).

**Missing values**: treat missing as its own bin/category `__MISSING__` for IG calculations.

**Small-sample stability**: if any bin has fewer than \(n_{min}=50\) samples (or <1%), merge with nearest neighbor (ordered bins) or into `__RARE__` bucket (unordered categories).

## 5) XGBoost objective + class imbalance handling

### 5.1 Objective (binary logistic)
Model a score function \(f(x)\) (additive boosted trees) and map to probability:
\[
\hat p_i = \sigma(f(x_i)) = \frac{1}{1+e^{-f(x_i)}}.
\]

Minimize a (possibly weighted) negative log-likelihood:
\[
\mathcal{L} = \sum_{i=1}^N w_i\,\ell(y_i, \hat p_i), \quad
\ell(y, \hat p) = -\big[y\log(\hat p) + (1-y)\log(1-\hat p)\big].
\]

### 5.2 Class imbalance (cost-sensitive weighting)
Defaults are typically rare, so use cost-sensitive weighting in training.

Let \(N_+\) be the number of positives and \(N_-\) negatives in the training set.

- **Default weight rule (placeholder)**:
  - positive weight \(w_+ = \frac{N_-}{N_+}\)
  - negative weight \(w_- = 1\)

This matches the common `scale_pos_weight \approx N_-/N_+` heuristic and is equivalent to maximizing a weighted log-likelihood.

**Rationale**: weighting reduces majority-class dominance in gradient updates. Since weights can distort probability calibration, calibration is treated as a first-class step (Section 5.3 / 6).

### 5.3 Probability calibration (post-model)
Tree ensembles can be miscalibrated. Apply a calibration mapping using only training/validation folds.

- **Primary**: isotonic regression calibration.
- **Alternative**: Platt scaling (logistic calibration) if isotonic overfits or the sample is too small.

**Leakage rule**: calibration must be trained on out-of-fold predictions (or a dedicated validation period) and never on the final test set.

## 6) Evaluation plan: discrimination + calibration + placeholders

### 6.1 Metrics
- **ROC-AUC**: ranking discrimination.
- **PR-AUC**: discrimination under class imbalance.
- **Brier score**:
\[
\text{Brier} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat p_i)^2.
\]
- **Calibration curve**: reliability diagram (e.g., deciles of \(\hat p\)) comparing mean predicted risk vs observed default rate.
- **(Optional)** log loss for probabilistic sharpness.

### 6.2 Acceptance criteria (placeholders; to be filled after first synthetic run)
These thresholds are intentionally placeholders until the synthetic DGP is executed once.

- **Discrimination**
  - ROC-AUC \(\ge\) `TBD_ROC_AUC`
  - PR-AUC \(\ge\) `TBD_PR_AUC`
- **Calibration**
  - Brier score \(\le\) `TBD_BRIER`
  - Calibration curve: max absolute deviation between predicted and observed in deciles \(\le\) `TBD_CAL_ERR`
- **Sanity**
  - Mean predicted \(\hat p\) matches empirical base rate within `TBD_BASE_RATE_TOL` **after calibration**.

### 6.3 Reporting slices (must)
Report all metrics overall and by key segments to detect hidden failure modes:
- `region`
- `is_newcomer`
- `months_on_book` bands
- optionally risk bands by `systemic_risk_index` (low/med/high)

## 7) Stability checks

### 7.1 Seed and split sensitivity
Run \(K\) repeats (e.g., \(K=10\)) varying random seeds for:
- train/validation split (if random split used for synthetic smoke-test)
- model stochasticity (subsampling, column sampling)

Report distribution (mean/std, min/max) for ROC-AUC, PR-AUC, Brier.

**Acceptance placeholder**: std-dev of ROC-AUC across seeds \(\le\) `TBD_AUC_STD`.

### 7.2 Time-split stability (when panel/longitudinal exists)
When longitudinal data exist, use rolling-origin evaluation:
- train on \([t_0, t_k]\), test on \((t_k, t_{k+1}]\)

### 7.3 Monotonicity constraints (optional, if used)
If monotonic constraints are applied, they must be domain-justified and tested.

Tentative monotone directions (only enforce if synthetic DGP and domain logic match):
- monotone increasing risk: `utilization_ratio`, `historical_missed_payments_12m`
- monotone decreasing risk: `income` (if present and DGP encodes it)

**Validation**: monotone check on held-out set must respect the constraint direction except for small tolerance `TBD_MONO_TOL`.

## 8) Leakage audit checklist (must-pass)
- **Temporal join**: macro row used for each \((i,t)\) satisfies \(\tau \le \tau(t)\).
- **Feature cutoff**: no customer behavior features computed from \((t, t+90\text{d}]\).
- **Preprocessing scope**: encoders/scalers/calibrators fit on training folds/time only.
- **Target leakage scan**: no feature name or derivation contains label-window proxies (e.g., “next_90_days”, “future”, “delinquency_after_t”).

---

## INTERFACE CONTRACT (v1)

1) Contract name:
- SystemicRiskIndexFeed

2) Owner + consumer(s):
- Owner agent: A
- Consuming agents: B, D, E

3) Purpose (1-2 sentences):
- Provide a leakage-safe macro-derived scalar `systemic_risk_index` for each macro timestamp to be consumed as an input feature in predictive scoring and downstream analytics.

4) Inputs (schema + units + timing):
- Input_1: name=macro_timeseries_df, type=DataFrame, unit=n/a, frequency=monthly, time_index=τ, allowable_range=n/a, missingness_policy=must document and provide imputation/ffill rules
- Input_2: name=as_of_macro_timestamp, type=date(or int month index), unit=n/a, frequency=monthly, time_index=τ, allowable_range=within available macro history, missingness_policy=reject if not in index

5) Outputs (schema + units + interpretation):
- Output_1: name=systemic_risk_index, type=float, unit=unitless, allowable_range=[0,1.5] (placeholder; must be finalized by Owner), interpretation=macro stress indicator usable as a predictive feature at time τ
- Output_2: name=systemic_risk_index_metadata, type=dict/json, unit=n/a, allowable_range=n/a, interpretation=includes exact definition version, computation timestamp, and valid_time_index range

6) Assumptions:
- Identification/causality assumptions (if any)
- Stationarity / regime assumptions
- Independence assumptions
- Any “only valid when …” conditions
- Only valid when computed using macro information available up to τ (no look-ahead)

7) Mathematical definition:
- Owner provides the explicit mapping, minimally: `systemic_risk_index_τ = g(p_regime_high_stress_τ, tariff_shock_multiplier_τ, …)` with all components defined and bounded

8) Validation & acceptance tests (must-pass):
- Test_1: For every τ, verify all inputs used to compute `systemic_risk_index_τ` have timestamps ≤ τ (pass/fail)
- Test_2: `systemic_risk_index` values fall within documented allowable_range; any out-of-range triggers failure or explicit clipping rule documented

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: add decomposition terms to support explainability (e.g., regime component vs shock component)

---

## INTERFACE CONTRACT (v1)

1) Contract name:
- PredictiveTrainingDatasetSchema

2) Owner + consumer(s):
- Owner agent: B
- Consuming agents: C, D, E, G

3) Purpose (1-2 sentences):
- Define the leakage-safe supervised learning table used for predictive model training and scoring, including the target `default_next_90_days`, feature columns, and time indices.

4) Inputs (schema + units + timing):
- Input_1: name=customer_snapshot_df, type=DataFrame, unit=n/a, frequency=event/as-of, time_index=t, allowable_range=n/a, missingness_policy=documented per column
- Input_2: name=macro_features_df, type=DataFrame, unit=n/a, frequency=monthly, time_index=τ, allowable_range=n/a, missingness_policy=documented per column
- Input_3: name=systemic_risk_index_df, type=DataFrame, unit=n/a, frequency=monthly, time_index=τ, allowable_range=n/a, missingness_policy=reject if missing for τ(t)

5) Outputs (schema + units + interpretation):
- Output_1: name=training_table_df, type=DataFrame, unit=n/a, allowable_range=n/a, interpretation=one row per (customer_id, as_of_time t) with features at ≤t and label over (t, t+90d]

6) Assumptions:
- Identification/causality assumptions (if any)
- Stationarity / regime assumptions
- Independence assumptions
- Any “only valid when …” conditions
- Only valid when as-of join uses τ(t) and label window is future-only

7) Mathematical definition:
- For each (i,t): features \(X_{i,t} = [X^{cust}_{i,t}, X^{macro}_{\tau(t)}, \texttt{systemic_risk_index}_{\tau(t)}]\); label \(Y_{i,t}=\mathbb{1}\{default \in (t,t+90d]\}\)

8) Validation & acceptance tests (must-pass):
- Test_1: Time-consistency audit: for sampled rows, verify all macro features use τ ≤ τ(t) and no features reference data after t
- Test_2: Schema audit: required columns present; dtypes and units match spec; `customer_id` not included in model feature matrix

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: add multiple as-of times per customer for panel training

---

## INTERFACE CONTRACT (v1)

1) Contract name:
- PredictiveRiskScoreOutput

2) Owner + consumer(s):
- Owner agent: B
- Consuming agents: D, E, app/dashboard (Lead Integrator), C

3) Purpose (1-2 sentences):
- Standardize the predictive model output as a calibrated probability of default over the next 90 days, enabling consistent downstream explainability, simulation, and treatment recommendation.

4) Inputs (schema + units + timing):
- Input_1: name=feature_row, type=dict/Series, unit=n/a, frequency=per scoring request, time_index=t, allowable_range=must match training schema, missingness_policy=apply training-time missing rules
- Input_2: name=calibration_version, type=string, unit=n/a, frequency=per model version, time_index=n/a, allowable_range=valid known versions, missingness_policy=default to current model version

5) Outputs (schema + units + interpretation):
- Output_1: name=pred_pd_90d, type=float, unit=probability, allowable_range=[0,1], interpretation=calibrated \(P(Y=1\mid X)\) for the next 90 days
- Output_2: name=pred_pd_90d_uncalibrated, type=float, unit=probability, allowable_range=[0,1], interpretation=raw model score before calibration
- Output_3: name=model_version, type=string, unit=n/a, allowable_range=n/a, interpretation=version identifier for reproducibility

6) Assumptions:
- Identification/causality assumptions (if any)
- Stationarity / regime assumptions
- Independence assumptions
- Any “only valid when …” conditions
- Only valid when input features follow the leakage-safe schema and calibration was fit without test leakage

7) Mathematical definition:
- \(\hat p = c(\sigma(f(x)))\) where \(f\) is the XGBoost ensemble score and \(c\) is the learned calibration mapping (identity if none)

8) Validation & acceptance tests (must-pass):
- Test_1: Output bounds: `pred_pd_90d` ∈ [0,1] for all scored rows
- Test_2: Calibration sanity: on validation set, average `pred_pd_90d` within each decile is within `TBD_CAL_ERR` of observed default rate

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: add confidence intervals via conformal prediction or bootstraps

