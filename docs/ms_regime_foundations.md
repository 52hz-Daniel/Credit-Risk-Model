# Markov-Switching Macro Regime Engine — Foundations + Validation Spec

## Purpose and scope
This document specifies the **macro regime (Markov-switching) engine** used to produce a bounded, interpretable time series:
- **`regime_prob_stress_t`**: probability that time \(t\) is a “stress” regime, based only on macro/alternative-data observables.
- **`systemic_risk_index_t`**: a dimensionless \([0,1]\) mapping of that probability, designed to be a stable downstream feature.

Hard constraints honored:
- **No code** in this deliverable (spec + validation checklist only).
- Uses only local references: **Hamilton (1989, 1990, 1996)** PDFs and `statsmodels` regime-switching reference implementation.

## 1) Model choices (MS-AR vs MS-VAR vs statsmodels terminology)

### 1.1 Terminology mapping (papers → `statsmodels`)
- **Hamilton (1989)**: a Markov-switching autoregressive specification; regime-dependent parameters; nonlinear **filter** and **smoother** for \(\Pr(S_t=\cdot\mid \mathcal{F}_t)\) and \(\Pr(S_t=\cdot\mid \mathcal{F}_T)\).
- **Hamilton (1990)**: general framework for Markov switching in **vector** systems and discussion of numerical issues; introduces an **EM** approach and emphasizes robustness issues (local maxima, boundary behavior).
- **Hamilton (1996)**: specification tests for Markov-switching models: omitted **autocorrelation**, omitted **ARCH**, misspecified **Markov dynamics**, omitted **explanatory variables**.

`statsmodels` implementation vocabulary (from `Reference Git Repo/statsmodels/statsmodels/tsa/regime_switching/markov_switching.py`):
- Probabilities (all are time-indexed arrays/series):
  - **`predicted_marginal_probabilities`**: \(\Pr(S_t\mid \mathcal{F}_{t-1})\)
  - **`filtered_marginal_probabilities`**: \(\Pr(S_t\mid \mathcal{F}_{t})\)
  - **`smoothed_marginal_probabilities`**: \(\Pr(S_t\mid \mathcal{F}_{T})\)
  - plus joint counterparts (`*_joint_probabilities`) when model order \(>0\).
- **`expected_durations`**: implied expected duration of each regime from the estimated transition matrix.

### 1.2 Primary model choice for this project (baseline)
Given the **short sample** implied by the master plan (e.g., ~60 monthly observations), the baseline must be parsimonious.

**Baseline**: **two-regime Markov-switching autoregression (MS-AR)** on a single macro stress indicator \(y_t\).
- In `statsmodels` terms, this aligns with the **Markov switching framework** and can be implemented via the `statsmodels.tsa.regime_switching.markov_autoregression` interface (project plan references it directly).

### 1.3 When to use a Markov-switching VAR (MS-VAR)
**Markov-switching VAR** is conceptually supported by Hamilton (1990) for \(\mathbf{y}_t\in\mathbb{R}^n\) (e.g., unemployment + inflation).
However, with short samples it is easy to over-parameterize.

**MS-VAR is permitted only if** all of the following are satisfied:
- **Identifiability**: regimes are well-separated (smoothed probabilities not diffuse for most \(t\); see validation).
- **Stability**: estimated regime assignments and transition probabilities are robust across multiple initializations.
- **Parsimony**: lag order and switching parameterization remain small enough to avoid near-singular covariance / unstable standard errors.

### 1.4 Switching variance (regime-dependent volatility): optional, guarded
Allowing regime-dependent variance can improve fit but introduces known numerical failure modes (Hamilton 1990 discusses singularities and boundary behavior when variances differ across regimes).

**Policy**:
- Default to **constant variance** across regimes unless diagnostics show strong omitted-ARCH patterns (Hamilton 1996).
- If switching variance is enabled, require additional mitigation/validation (Section 6).

## 2) Formal notation (state process, observation equations, likelihood/filter/smoother outputs)

### 2.1 Time index and information set
- Time is discrete monthly: \(t=1,\dots,T\).
- \(\mathcal{F}_t\) denotes all observed information through time \(t\) (macro/alt-data series up to \(t\)).

### 2.2 Hidden state (regime) process
Baseline: \(K=2\) regimes.

- Hidden regime variable \(S_t \in \{0,1\}\).
- First-order Markov dynamics with transition matrix \(P\):
\[
p_{ij} \equiv \Pr(S_t=i \mid S_{t-1}=j),\quad
P =
\begin{pmatrix}
\Pr(S_t=0\mid S_{t-1}=0) & \Pr(S_t=0\mid S_{t-1}=1)\\
\Pr(S_t=1\mid S_{t-1}=0) & \Pr(S_t=1\mid S_{t-1}=1)
\end{pmatrix}
\]
where each column sums to 1 (left-stochastic convention, matching `statsmodels`).

### 2.3 Observation equation(s)

#### Option A (baseline): univariate MS-AR(p)
Let \(y_t\) be a **single macro stress indicator** (constructed only from macro/alt-data, not retail defaults).

\[
y_t = c_{S_t} + \sum_{k=1}^{p} \phi_{k,S_t}\,y_{t-k} + \varepsilon_t,\qquad
\varepsilon_t \sim \mathcal{N}(0,\sigma^2)\ \text{(baseline)}
\]
Optional switching variance variant:
\[
\varepsilon_t \sim \mathcal{N}(0,\sigma^2_{S_t})
\]

#### Option B (conceptual extension): Markov-switching VAR(p)
Let \(\mathbf{y}_t\in\mathbb{R}^n\) be a vector of macro observables:
\[
\mathbf{y}_t = \mathbf{c}_{S_t} + \sum_{k=1}^{p} A_{k,S_t}\,\mathbf{y}_{t-k} + \mathbf{u}_t,\qquad
\mathbf{u}_t \sim \mathcal{N}(\mathbf{0},\Sigma_{S_t}\ \text{or}\ \Sigma)
\]
This extension must follow the MS-VAR guardrails in Section 1.3.

### 2.4 Likelihood objects and filter/smoother outputs (aligned to `statsmodels`)
Define conditional density (abstractly):
\[
f(y_t\mid S_t, \text{lags}, \theta)
\]

The regime inference objects required for this project (and provided by `statsmodels`) are:
- **Predicted marginal probabilities**:
\[
\Pr(S_t=i\mid \mathcal{F}_{t-1}) \quad \leftrightarrow \quad \texttt{predicted\_marginal\_probabilities}[t,i]
\]
- **Filtered marginal probabilities**:
\[
\Pr(S_t=i\mid \mathcal{F}_{t}) \quad \leftrightarrow \quad \texttt{filtered\_marginal\_probabilities}[t,i]
\]
- **Smoothed marginal probabilities**:
\[
\Pr(S_t=i\mid \mathcal{F}_{T}) \quad \leftrightarrow \quad \texttt{smoothed\_marginal\_probabilities}[t,i]
\]
- **Expected durations** (per regime) implied by \(P\):
for regime \(i\), \( \mathbb{E}[\text{duration}\mid S_t=i] = \frac{1}{1-p_{ii}}\) when non-degenerate, matching `statsmodels.expected_durations`.

**Probability convention for “current risk”**:
- For real-time signal at the most recent month \(T\), use **filtered** probability: \(\Pr(S_T=\text{stress}\mid \mathcal{F}_T)\).
- Use **smoothed** probabilities only for retrospective analysis and offline validation plots.

## 3) Regime meaning: labeling the “stress regime” without circularity

### 3.1 Non-circularity requirement
The label “stress regime” must be assigned using only **macro/alt-data behavior**. It must **not** be defined using:
- retail performance (defaults, delinquencies),
- downstream predictive model outcomes,
- any variable mechanically constructed from `systemic_risk_index`.

### 3.2 Stress-regime labeling rule (deterministic)
Because regime labels are exchangeable (Hamilton 1989 notes the state numbering is arbitrary), define a deterministic labeling rule:

Let \(g(\cdot)\) be the constructed scalar macro stress indicator used as \(y_t\).

**Primary rule**:
- Label the regime with **higher conditional mean** \(\mu_i\) of \(y_t\) as **stress**:
  - stress \(=\arg\max_{i\in\{0,1\}} \mu_i\)

**Tie-breakers** (used only if primary rule is ambiguous / statistically indistinguishable):
- If switching variance is used, label stress as higher variance regime:
  - stress \(=\arg\max_i \sigma_i^2\)
- If still ambiguous, label stress as regime with higher implied **expected duration** during historically known stress-like macro episodes (defined from macro observables only), but keep the rule explicit and reproducible.

### 3.3 Interpretability requirements
The final labeled “stress” regime must satisfy (at least) one macro-based interpretation check:
- The stress regime corresponds to systematically worse macro conditions **in the inputs** (e.g., higher unemployment, higher inflation, lower growth proxy, worse sentiment, higher social stress), depending on what feeds \(y_t\).

## 4) Output contract: `regime_prob_stress_t` and mapping to `systemic_risk_index`

### 4.1 `regime_prob_stress_t`
Definition:
\[
\texttt{regime\_prob\_stress\_t} \equiv \Pr(S_t=\text{stress}\mid \mathcal{F}_t)
\]
Implementation alignment (future code): use `filtered_marginal_probabilities` and select the column corresponding to the deterministically labeled stress regime.

Required properties:
- **Bounded**: \(0\le \texttt{regime\_prob\_stress\_t}\le 1\).
- **Time index**: monthly, aligned to macro dataset index.
- **No look-ahead** for the latest point: the “current” value must be filtered, not smoothed.

### 4.2 `systemic_risk_index_t` mapping
Baseline mapping (must be monotone, bounded, and dimensionless):
\[
\texttt{systemic\_risk\_index\_t} \equiv \mathrm{clamp}(\texttt{regime\_prob\_stress\_t}, 0, 1)
\]

Optional calibration (allowed only if documented and monotone):
- affine + clamp: \(\mathrm{clamp}(a\cdot p_t + b, 0, 1)\), \(a>0\)
- logistic transform of a centered probability (monotone), then clamp.

**Units and bounds**:
- Unit: **dimensionless**
- Allowable range: **\([0,1]\)** inclusive

### 4.3 Relationship to downstream modules
Downstream models must treat:
- `regime_prob_stress_t` as a probabilistic regime indicator (more volatile).
- `systemic_risk_index_t` as the stabilized, bounded macro systemic-risk feature.

## 5) Diagnostics & validation checklist (Hamilton 1996-derived)

This section is a **must-run checklist**. The categories match Hamilton (1996): omitted autocorrelation, omitted ARCH, Markov dynamics misspecification, omitted explanatory variables.

### 5.1 Basic sanity checks (pre-Hamilton-1996)
- **Probability sanity**:
  - All regime probabilities are in \([0,1]\) and rows sum to 1 (numerically within tolerance).
- **Regime persistence sanity**:
  - Estimated \(p_{00}\) and \(p_{11}\) are interior (not exactly 0 or 1), unless there is strong evidence of a degenerate regime; check `expected_durations`.
- **Regime separation**:
  - Smoothed probabilities are not uniformly ~0.5 for all \(t\) (indicates non-identification in short samples).

### 5.2 Omitted autocorrelation (Hamilton 1996)
Goal: detect whether the model left serial correlation unmodeled (either within regimes or across regimes).

Must checks:
- **Residual ACF/PACF** on \(e_t = y_t - \hat{y}_t\) overall.
- **Regime-weighted residual autocorrelation**:
  - Use smoothed regime weights \(w_{t,i}=\Pr(S_t=i\mid \mathcal{F}_T)\) and check autocorrelation structure within each regime.

Pass/Fail guidance:
- Fail if there is statistically and economically meaningful residual autocorrelation at low lags that persists across plausible specifications (e.g., increasing \(p\) from 1 to 2).

### 5.3 Omitted ARCH / heteroskedasticity (Hamilton 1996)
Goal: detect whether conditional variance dynamics are missing.

Must checks:
- ARCH-type diagnostics on squared residuals \(e_t^2\).
- Regime-weighted variance checks (if variance is not switching):
  - verify residual variance does not cluster systematically in one regime beyond what the model implies.

Mitigation if failed:
- enable switching variance \(\sigma^2_{S_t}\) **only** with singularity safeguards (Section 6.5), or add limited variance dynamics in a controlled way.

### 5.4 Misspecification of Markov dynamics (Hamilton 1996)
Goal: validate the assumption that the hidden state follows a **first-order** Markov chain independent of other information.

Must checks:
- **First-order vs higher-order** Markov adequacy:
  - test whether state persistence appears to depend on \(S_{t-2}\) beyond \(S_{t-1}\) (Hamilton 1996 “validity of Markov assumptions” category).
- **Independence from observables**:
  - check whether transition behavior appears to depend on macro covariates not included in the transition mechanism (if later using time-varying transition probabilities, TVTP).

Pass/Fail guidance:
- Fail if transition dynamics imply systematically predictable deviations (e.g., many improbable quick flips) that are stable across initializations.

### 5.5 Omitted explanatory variables (Hamilton 1996)
Goal: detect whether additional observed variables \(z_t\) should enter the **mean** or **variance** equation.

Must checks:
- Candidate omitted variables list (macro-only): inflation surprise proxy, unemployment change, sentiment shifts, social stress index changes, etc.
- Test whether adding \(z_t\) materially reduces residual autocorrelation / ARCH patterns and improves regime interpretability without destroying parsimony.

Pass/Fail guidance:
- Fail if residual structure strongly indicates missing explanatory power and regimes become uninterpretable without key macro covariates.

### 5.6 Small-sample caution (Hamilton 1996)
Hamilton (1996) reports small-sample behavior where asymptotic \(\chi^2\) approximations can be too stringent.

Policy for this project (short samples, e.g., \(T\approx 60\)):
- Treat asymptotic results as **screening**; require robustness across:
  - alternative initializations,
  - slightly altered lag orders / parsimonious variants,
  - and stable regime labeling.

## 6) Failure modes + mitigations (implementation-agnostic)

### 6.1 Label switching / regime relabeling
**Failure mode**: the optimizer returns regimes swapped between runs.

Mitigation:
- Apply the deterministic stress-labeling rule in Section 3.2 after estimation.
- Enforce an ordering constraint conceptually (e.g., \(\mu_{\text{stress}}>\mu_{\text{calm}}\)) when possible.

### 6.2 Local maxima / unstable likelihood surface (Hamilton 1990)
**Failure mode**: multiple local maxima, dependence on starting values.

Mitigation:
- Multi-start estimation conceptually; select solution via:
  - highest log-likelihood among acceptable solutions,
  - plus interpretability and diagnostics (Section 5),
  - plus non-degenerate transition probabilities.
- Prefer EM-style robust initialization steps (Hamilton 1990 rationale), then refine.

### 6.3 Short sample / parameter explosion
**Failure mode**: MS-VAR or high-order MS-AR overfits; probabilities become diffuse; transition probs hit boundaries.

Mitigation:
- Default to MS-AR(1) or MS-AR(2).
- Keep \(K=2\).
- Restrict which parameters switch (start with switching mean only; then consider switching AR/variance if diagnostics demand).
- Use information criteria (AIC/BIC/HQIC) as secondary, not primary, decision tools in short samples.

### 6.4 Degenerate transitions / near-absorbing states
**Failure mode**: \(p_{ii}\approx 1\) leading to extremely long expected durations; or \(p_{ii}\approx 0\) leading to implausible flipping.

Mitigation:
- Require `expected_durations` to be plausible for monthly macro regimes (project-specific acceptable bands set in acceptance tests below).
- If degeneracy is stable across starts, consider:
  - alternative \(y_t\) construction (more signal),
  - fewer switching parameters,
  - or explicit constraints/prior-like regularization in future implementation.

### 6.5 Switching variance singularities (Hamilton 1990)
**Failure mode**: variance collapses toward 0 for a regime that “captures” a small subset of points (likelihood singularity).

Mitigation policy if switching variance is used:
- Impose lower bounds on \(\sigma_i^2\) in implementation.
- Reject solutions where any \(\sigma_i^2\) is implausibly small relative to empirical variance of \(y_t\).
- Require robustness across re-starts; if only one start finds the solution, treat as suspect.

## 7) Interface contracts (exact template)

### INTERFACE CONTRACT (v1)

1) Contract name:
- MacroDataMonthlyInputs

2) Owner + consumer(s):
- Owner agent: <F>
- Consuming agents: <A>

3) Purpose (1-2 sentences):
- Provide a single, validated monthly macro/alternative-data table used to construct the macro stress indicator \(y_t\) and to align the Markov-switching time index.

4) Inputs (schema + units + timing):
- Input_1: name=<macro_df>, type=<pandas.DataFrame>, unit=<mixed by column; see allowable_range>, frequency=<monthly>, time_index=<t>, allowable_range=<time_index strictly increasing; no duplicate months>, missingness_policy=<no missing in core columns after preprocessing; allowed missing in optional columns if documented and imputed upstream>
- Input_2: name=<core_columns>, type=<list[str]>, unit=<n/a>, frequency=<static>, time_index=<n/a>, allowable_range=<must include at least unemployment and inflation proxies>, missingness_policy=<must be present; fail fast if absent>

5) Outputs (schema + units + interpretation):
- Output_1: name=<macro_df_validated>, type=<pandas.DataFrame>, unit=<mixed>, allowable_range=<aligned monthly index; all core columns finite>, interpretation=<canonical macro dataset passed to regime engine>

6) Assumptions:
- Identification/causality assumptions (if any)
- Stationarity / regime assumptions
- Independence assumptions
- Any “only valid when …” conditions
- Only valid when macro series are consistently transformed to comparable monthly frequency (e.g., aggregation rules are stable over time).

7) Mathematical definition:
- Provide the key equations or algorithm steps (minimum needed to be unambiguous)
- Time alignment: \(t\) corresponds to a month-end bucket; each observation is the value for that month after agreed aggregation (e.g., monthly average or end-of-month), applied consistently for all columns.

8) Validation & acceptance tests (must-pass):
- Test_1: core columns present and numeric; pass if all required columns exist and contain finite values for all \(t\).
- Test_2: time index monotone and unique; pass if index is strictly increasing with no duplicates and frequency is monthly (or monthly-parseable).

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: <optional>

### INTERFACE CONTRACT (v1)

1) Contract name:
- GameTheoryRiskMultiplier

2) Owner + consumer(s):
- Owner agent: <G>
- Consuming agents: <A>

3) Purpose (1-2 sentences):
- Provide a bounded, dimensionless multiplier representing geopolitical/tariff-game stress, to be combined with the macro regime probability in a way that preserves \([0,1]\) bounds.

4) Inputs (schema + units + timing):
- Input_1: name=<time_index_t>, type=<datetime-like monthly index>, unit=<n/a>, frequency=<monthly>, time_index=<t>, allowable_range=<must align to macro_df_validated index>, missingness_policy=<no missing for current month; forward-fill allowed only if explicitly flagged>

5) Outputs (schema + units + interpretation):
- Output_1: name=<tariff_shock_multiplier_t>, type=<float>, unit=<dimensionless>, allowable_range=<[1.0, 1.5]>, interpretation=<multiplicative stress factor from tariff-game equilibrium; 1.0 means neutral, higher means more stress>

6) Assumptions:
- Identification/causality assumptions (if any)
- Stationarity / regime assumptions
- Independence assumptions
- Any “only valid when …” conditions
- Only valid when multiplier construction does not use retail-default outcomes and is anchored to exogenous sentiment/political inputs.

7) Mathematical definition:
- Provide the key equations or algorithm steps (minimum needed to be unambiguous)
- The multiplier is an exogenous scalar \(m_t \in [1.0,1.5]\) derived from the game-theory module at month \(t\).

8) Validation & acceptance tests (must-pass):
- Test_1: bounds check; pass if \(1.0 \le m_t \le 1.5\) for all \(t\).
- Test_2: alignment check; pass if multiplier index matches macro index exactly (same months, same ordering).

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: <optional>

### INTERFACE CONTRACT (v1)

1) Contract name:
- MacroRegimeOutputsToSystemicRiskIndex

2) Owner + consumer(s):
- Owner agent: <A>
- Consuming agents: <B, C, E, G, D>

3) Purpose (1-2 sentences):
- Provide stable, bounded macro regime outputs for downstream feature engineering, simulation, explainability, and dashboard visualization.

4) Inputs (schema + units + timing):
- Input_1: name=<regime_prob_stress_t>, type=<float>, unit=<probability>, frequency=<monthly>, time_index=<t>, allowable_range=<[0,1]>, missingness_policy=<no missing; if model fails, return explicit failure state upstream rather than NaN>
- Input_2: name=<tariff_shock_multiplier_t>, type=<float>, unit=<dimensionless>, frequency=<monthly>, time_index=<t>, allowable_range=<[1.0, 1.5]>, missingness_policy=<no missing for current t; see GameTheoryRiskMultiplier>

5) Outputs (schema + units + interpretation):
- Output_1: name=<systemic_risk_index_t>, type=<float>, unit=<dimensionless>, allowable_range=<[0,1]>, interpretation=<bounded systemic risk feature; monotone in regime_prob_stress_t; preserves comparability across runs>
- Output_2: name=<regime_label_stress>, type=<int>, unit=<n/a>, allowable_range=<{0,1}>, interpretation=<which latent regime index was labeled as stress by deterministic rule>

6) Assumptions:
- Identification/causality assumptions (if any)
- Stationarity / regime assumptions
- Independence assumptions
- Any “only valid when …” conditions
- Only valid when stress regime labeling is determined from macro-only parameters (Section 3) and not from retail outcomes.

7) Mathematical definition:
- Provide the key equations or algorithm steps (minimum needed to be unambiguous)
- \(p_t = \Pr(S_t=\text{stress}\mid \mathcal{F}_t)\).
- Base index: \(\texttt{systemic\_risk\_index\_t} = \mathrm{clamp}(p_t, 0, 1)\).
- If multiplier is incorporated later (per master plan concept), the combination rule must preserve bounds, e.g.:
  - \(\texttt{systemic\_risk\_index\_t} = \mathrm{clamp}(p_t \cdot m_t, 0, 1)\), with \(m_t\in[1.0,1.5]\).

8) Validation & acceptance tests (must-pass):
- Test_1: bounds and monotonicity; pass if index stays in \([0,1]\) and increases when \(p_t\) increases holding other inputs fixed.
- Test_2: non-circular labeling; pass if no retail outcome fields are used in determining `regime_label_stress` or `regime_prob_stress_t`.

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: <optional>

## Validation checklist (summary, must-pass)
Use this as a release gate for the macro regime engine implementation.

- **Model choice**:
  - Baseline MS-AR(1) or MS-AR(2) used unless MS-VAR guardrails pass.
- **Stress labeling**:
  - Deterministic, macro-only rule applied; stable under relabeling.
- **Outputs**:
  - `regime_prob_stress_t` in \([0,1]\), filtered for current month.
  - `systemic_risk_index_t` in \([0,1]\), monotone in probability; if multiplier used, clamp preserves bounds.
- **Hamilton (1996) diagnostics**:
  - No material omitted autocorrelation remaining in residuals.
  - No material omitted ARCH remaining; or switching variance justified and stable.
  - Markov dynamics plausible (no strong evidence requiring higher order) and stable across starts.
  - No strong evidence of omitted macro explanatory variables needed for mean/variance.
- **Robustness**:
  - Results (probabilities, transition matrix, regime labeling) stable across multiple initializations; no dependence on a single fragile optimum.

