# Causal Inference / Uplift Spec (Step‑1)

## 0) Scope and objectives
This document specifies the **intervention engine** that estimates heterogeneous causal effects (CATE/uplift) of credit actions on short-horizon default risk, and defines **identification assumptions**, **what breaks them**, and a **validation plan** suitable for model risk review.

Hard constraints for Step‑1:
- **No implementation/code** in this step (spec only).
- **Multi‑arm intervention** aligned to shared vocabulary: `treatment_group ∈ {0,1,2}`.
- Explicitly document identification assumptions and failure modes.

---

## 1) Treatments and outcome definitions

### 1.1 Treatment definition (multi-arm)
We define a discrete action \(T \in \{0,1,2\}\) applied at decision time \(t\).

- **\(T=0\) Control / No Action**: do nothing; maintain current terms.
- **\(T=1\) Limit Cut 20%**: reduce revolving credit limit by 20% (relative change); applied immediately at \(t\).
- **\(T=2\) Payment Holiday**: grant a temporary payment holiday (e.g., minimum payment waived for a defined window) starting at \(t\).

**Timing rule (pre-treatment features):**
All covariates \(X\) used for causal estimation must be measured **strictly before** the action at \(t\). Any feature influenced by the action (post‑treatment) must not enter \(X\).

### 1.2 Outcome definition
Primary outcome:
- **`default_next_90_days`**: binary outcome \(Y \in \{0,1\}\) indicating whether the customer defaults in the **90 days after** decision time \(t\).

Secondary outcomes (optional for later phases, not required for Step‑1 validity):
- delinquency states (30/60/90 DPD), utilization change, attrition/closure within 90d, and expected margin (policy optimization).

---

## 2) Causal estimands and model outputs

### 2.1 Potential outcomes and CATE (multi-arm)
Let \(Y(t)\) denote potential outcome under treatment \(t \in \{0,1,2\}\).

For any two arms \(a\) and \(b\), define the **pairwise CATE**:
\[
\tau_{a,b}(x) := \mathbb{E}\left[ Y(a) - Y(b) \mid X=x \right]
\]
Key pairwise effects we will estimate and report:
- \(\tau_{1,0}(x)\): effect of **Limit Cut** vs Control
- \(\tau_{2,0}(x)\): effect of **Payment Holiday** vs Control
- \(\tau_{2,1}(x)\): effect of **Payment Holiday** vs Limit Cut (optional if needed for strategy tie‑breaks)

Interpretation (binary outcome):
- \(\tau_{a,b}(x)\) is an **absolute risk difference** in default probability (percentage‑point change), conditional on \(X=x\).

### 2.2 Uplift definition
For treatment \(a\) relative to control:
\[
\text{uplift}_a(x) := -\tau_{a,0}(x)
\]
So positive uplift means **default risk decreases** under treatment \(a\) vs control (beneficial).

### 2.3 Policy and policy value
A decision policy \(\pi(x)\in\{0,1,2\}\) maps features to an action.

Define the policy value (risk-only objective for Step‑1):
\[
V(\pi) := \mathbb{E}\left[ Y(\pi(X)) \right]
\]
A lower \(V(\pi)\) is better if minimizing defaults. In later steps, replace \(Y\) with a utility \(U\) incorporating margin, attrition, and constraints.

### 2.4 Required model outputs
At minimum, the intervention engine must output, for each customer row \(i\) with features \(x_i\):
- \(\widehat{\tau}_{1,0}(x_i)\) and \(\widehat{\tau}_{2,0}(x_i)\) (risk differences)
- A recommended action \(\widehat{\pi}(x_i)\) under a stated objective (risk minimization for Step‑1)
- Diagnostics needed for validation (propensity scores/overlap stats; uncertainty proxy optional but recommended)

---

## 3) Identification assumptions (and what breaks them)

### 3.1 SUTVA (Stable Unit Treatment Value Assumption)
Assumption:
- **No interference**: one customer’s treatment does not change another customer’s outcome.
- **No hidden versions of treatment**: “Limit cut 20%” and “Payment holiday” are well-defined and consistently applied.

What breaks it:
- Portfolio-level policy changes that affect collections capacity, call center outreach, or macro exposures in ways that couple customers.
- Interactions via shared household accounts, co-borrowers, or network effects (e.g., fraud rings).
- Operational heterogeneity: “payment holiday” varies in duration/eligibility/fees but is recorded as the same \(T=2\).

Mitigations (spec-level):
- Define treatment operationally with a clear configuration (duration, eligibility rules).
- If multiple versions exist, split into separate treatment arms or encode dosage and use continuous / multi-valued treatment methods.

### 3.2 Ignorability / Unconfoundedness (selection on observables)
Assumption:
\[
\{Y(0),Y(1),Y(2)\} \perp T \mid X
\]
Meaning: conditional on observed pre-treatment covariates \(X\), treatment assignment is “as good as random.”

What breaks it:
- **Human-in-the-loop discretion**: agents apply holidays based on soft information (tone on calls, hardship narratives) not in \(X\).
- **Reactive intervention** triggered by latent early-warning signals not captured (e.g., internal alerts, unresolved disputes, missed-but-not-posted payments).
- **Post-treatment leakage**: features used in \(X\) include variables affected by treatment (e.g., balance after limit cut).
- **Simulated data**: if synthetic assignment is randomized, ignorability holds by construction; if not, it may be violated.

Mitigations:
- Expand \(X\) to include the true assignment drivers (eligibility flags, hardship indicators, operational channel).
- Enforce strict time-indexing and feature provenance to avoid post-treatment variables.
- Use sensitivity analysis to quantify how strong an unobserved confounder would need to be to overturn conclusions.

### 3.3 Overlap / Positivity
Assumption:
For all \(x\) in the analysis support, propensity is bounded away from 0:
\[
0 < \mathbb{P}(T=t \mid X=x) < 1,\quad \forall t\in\{0,1,2\}
\]

What breaks it:
- Deterministic rules (e.g., “holiday only for newcomers”) producing near-zero propensity in large regions of \(X\).
- Rare treatments or extreme targeting (only the highest-risk get a holiday).
- Dataset shift where deployment population differs from training support.

Mitigations:
- Diagnose overlap (propensity histograms; effective sample size; trimming rules).
- Restrict estimands to a common support region (explicit “eligible population”).
- Prefer estimators robust to mild overlap issues and report uncertainty/instability.

### 3.4 Consistency and well-defined time horizon
Assumption:
- Observed outcome equals the potential outcome under the received treatment, with a consistent 90-day measurement window.

What breaks it:
- Censoring/attrition (account closure, charge-off definition changes) within 90 days.
- Treatment timing ambiguity (holiday starts mid-cycle; limit cut delayed).
- Label leakage (default definition depends on treatment, e.g., holiday changes delinquency labeling).

Mitigations:
- Define outcome operationally independent of treatment mechanics where possible.
- Use consistent observation window and handle censoring explicitly (later phase).

---

## 4) Estimator family recommendation (first implementation choice)

### Recommendation for Step‑1: **EconML doubly-robust learners (DR/DML) first**
Preferred first estimator family:
- **Doubly Robust (DR) / Double Machine Learning (DML)** style CATE estimators (e.g., DR-learner / R-learner variants) implemented in **EconML**.

Why (relative to CausalML X‑learner as first choice):
- **Robustness**: DR-style methods provide consistency if *either* the propensity model \(e_t(x)\) *or* the outcome model \(\mu_t(x)\) is well-specified (doubly robust intuition), and behave better under moderate misspecification.
- **Validation hooks**: they naturally yield **pseudo-outcomes** and residualization checks that make Step‑1 validation clearer and more model-risk-friendly.
- **Multi-arm extensibility**: DR framing extends to multiple treatments via generalized propensity and nuisance models.

When to prefer **CausalML X‑learner** early (secondary baseline):
- If treatments are imbalanced and you want a strong meta-learner baseline quickly.
- If you want to align tightly with the project’s planned “X‑learner + XGBoost base learner” path (see Phase 7 of the master plan). In that case, run X‑learner as a benchmark but retain DR-style checks as acceptance gates.

Practical approach in Step‑1:
- Specify **both** as candidates, with acceptance criteria driven by overlap diagnostics and pseudo-outcome sanity; choose the one that passes validation most reliably on the current dataset.

---

## 5) Validation plan (must-pass checks)

### 5.1 Data and design validation (pre-model)
- **Time indexing audit**: confirm all \(X\) precedes \(T\); no post-treatment leakage.
- **Treatment integrity**: verify `treatment_group` values map unambiguously to actions.
- **Overlap diagnostics**:
  - For each arm \(t\), inspect \(\hat e_t(x)\) distribution; flag near-0/near-1 mass.
  - Define a trimming rule (e.g., drop or downweight samples with \(\max_t \hat e_t(x) > 0.98\) or \(\min_t \hat e_t(x) < 0.02\)) and show sensitivity of estimates to trimming.

### 5.2 Nuisance model sanity (propensity and outcome)
- **Propensity sanity**: propensity model should predict treatment better than chance but not perfectly; extreme AUC may indicate deterministic rules (overlap failure) or leakage.
- **Outcome model sanity**: predict \(Y\) within each arm; check calibration (Brier score / calibration curve) and monotonic feature effects sanity.

### 5.3 Pseudo-outcome checks (core causal validation)
For DR/DML-type estimators:
- Compute pseudo-outcomes (e.g., orthogonalized scores). Must-pass:
  - **Finite, stable distribution** (no heavy tails driven by extreme \(1/\hat e_t(x)\)).
  - **Mean alignment**: averaged CATE over population should align with ATE estimates from a simpler DR ATE estimator (within tolerance).

For X-learner:
- Must-pass:
  - Imputed counterfactual outcomes are in valid range \([0,1]\) for binary outcomes (or are properly transformed back).
  - Treatment-effect model residuals are not dominated by a handful of high-leverage points.

### 5.4 Doubly-robust “sanity” comparisons
- Compare:
  - Outcome-regression-only estimate (g-formula style)
  - IPW-only estimate
  - DR estimate
If IPW and outcome regression disagree substantially but DR is stable, that is acceptable; if DR swings wildly with small propensity changes, overlap is likely poor.

### 5.5 Placebo / negative-control tests
- **Placebo outcome**: choose an outcome known to be unaffected by the treatment (e.g., a pre-treatment lagged default indicator, or a stable attribute). Estimated CATE should be ~0.
- **Placebo treatment**: permute treatment labels within propensity strata; estimated CATE should be ~0 and policy value should not improve beyond noise.

### 5.6 Sensitivity to propensity specification
- Refit propensity with different model classes / regularization; re-estimate CATE.
- Must-pass:
  - Rank correlation of \(\widehat{\tau}_{t,0}(x)\) across specifications exceeds a threshold (project-defined), or else the model is deemed too fragile for prescription.

### 5.7 Policy value validation (offline)
Evaluate candidate policy \(\hat\pi\) using off-policy estimators:
- **IPS** and **DR** policy value estimates for \(V(\pi)\) (risk-only for Step‑1).
Must-pass:
- Policy value improvements persist under DR evaluation and do not disappear with mild trimming changes.

---

## 6) Failure modes and how we detect them

- **Selection bias / unobserved confounding**
  - Symptom: strong uplift in segments tied to discretionary operations not captured in \(X\); placebo tests fail.
  - Response: add missing covariates, narrow eligible population, sensitivity analysis.

- **Poor overlap / positivity violations**
  - Symptom: extreme propensities, unstable pseudo-outcomes, large variance, heavy tails.
  - Response: trimming, redefine population, or avoid prescribing actions where support is missing.

- **Extrapolation (out-of-support ITEs)**
  - Symptom: very large \(|\widehat{\tau}|\) in sparse regions; high model uncertainty proxies.
  - Response: constrain prescriptions to support; add conservative caps and abstain rules (later phase).

- **Unstable ITEs / overfitting**
  - Symptom: CATE highly sensitive to nuisance specs, folds, or random seeds; low rank stability.
  - Response: stronger regularization, cross-fitting, simpler models, or focus on group-level CATE first.

- **Post-treatment leakage**
  - Symptom: implausibly strong treatment predictability; outcome model uses variables affected by treatment.
  - Response: strict feature timestamping; rebuild feature set.

- **Interference / operational spillovers**
  - Symptom: treatment effects vary with portfolio-level volumes rather than customer covariates.
  - Response: redesign study, incorporate cluster/time effects, or move to experimentation.

---

## 7) Exactly 3 interface contracts (template: INTERFACE CONTRACT v1)

INTERFACE CONTRACT (v1)

1) Contract name:
- PredictiveFeatureBundleForCausal_v1

2) Owner + consumer(s):
- Owner agent: B
- Consuming agents: C

3) Purpose (1-2 sentences):
- Provide a strictly pre-treatment feature matrix \(X\) (and optional baseline risk score) that the causal engine uses for propensity/outcome nuisance models and CATE estimation, with clear timing and leakage constraints.

4) Inputs (schema + units + timing):
- Input_1: name=customer_features_X, type=DataFrame[float|int|category], unit=mixed, frequency=per_decision, time_index=t, allowable_range=feature-specific, missingness_policy=impute_or_flag_pre_treatment_only
- Input_2: name=baseline_pd_hat, type=float, unit=probability, frequency=per_decision, time_index=t, allowable_range=[0,1], missingness_policy=required_or_recompute
- Input_3: name=feature_timestamp_map, type=Dict[str, datetime|int], unit=timestamp, frequency=per_training_batch, time_index=t, allowable_range=n/a, missingness_policy=required

5) Outputs (schema + units + interpretation):
- Output_1: name=leakage_check_passed, type=bool, unit=flag, allowable_range={True,False}, interpretation=True means all features are verified as pre-treatment at t
- Output_2: name=eligible_population_mask, type=bool_array, unit=flag, allowable_range={True,False}, interpretation=rows where treatment/outcome definitions and timing are valid for causal estimation

6) Assumptions:
- Identification/causality assumptions (if any)
  - Features in customer_features_X are not affected by treatment assignment at time t (no post-treatment leakage).
- Stationarity / regime assumptions
  - Feature definitions remain stable across training and scoring windows.
- Independence assumptions
  - Row-wise IID conditional on provided time_indexing; clustering handled later if needed.
- Any “only valid when …” conditions
  - Only valid when decision time t is well-defined and aligned across treatment and outcome windows.

7) Mathematical definition:
- Let \(X\) be the provided feature vector at time \(t\). Contract ensures \(X \subseteq \mathcal{F}_{t^-}\) (pre-treatment sigma-field).
- Baseline PD \(\hat p(X)\) is an optional additional covariate; it must also be computed from pre-treatment information only.

8) Validation & acceptance tests (must-pass):
- Test_1: Feature timing audit — for a random sample of rows/features, verify timestamp(feature) < t; pass if 100% of checked features satisfy.
- Test_2: Leakage sentinel — train a classifier to predict T from X; pass if performance is plausible and not near-deterministic absent explicit deterministic eligibility rules (threshold defined by integrator).

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: add clustered identifiers (branch_id, agent_id) and time-window features with explicit lag policies


INTERFACE CONTRACT (v1)

1) Contract name:
- InterventionCATEOutputsToStrategy_v1

2) Owner + consumer(s):
- Owner agent: C
- Consuming agents: G, E, app/dashboard (via integrator)

3) Purpose (1-2 sentences):
- Provide per-customer estimated causal effects for each action (vs control), plus a recommended action under a stated objective, for downstream strategy optimization and simulation.

4) Inputs (schema + units + timing):
- Input_1: name=customer_features_X, type=DataFrame[float|int|category], unit=mixed, frequency=per_decision, time_index=t, allowable_range=feature-specific, missingness_policy=must_match_training_schema
- Input_2: name=objective, type=enum, unit=n/a, frequency=per_run, time_index=t, allowable_range={minimize_default_risk}, missingness_policy=default=minimize_default_risk
- Input_3: name=action_constraints, type=Dict, unit=n/a, frequency=per_run, time_index=t, allowable_range=n/a, missingness_policy=optional

5) Outputs (schema + units + interpretation):
- Output_1: name=cate_limit_cut_vs_control, type=float, unit=probability_points, allowable_range=[-1,1], interpretation=estimated \(E[Y(1)-Y(0)|X]\); negative means reduces default risk
- Output_2: name=cate_holiday_vs_control, type=float, unit=probability_points, allowable_range=[-1,1], interpretation=estimated \(E[Y(2)-Y(0)|X]\); negative means reduces default risk
- Output_3: name=recommended_action, type=int, unit=arm_id, allowable_range={0,1,2}, interpretation=argmin expected default risk under constraints

6) Assumptions:
- Identification/causality assumptions (if any)
  - SUTVA, ignorability given X, and overlap on the eligible population.
- Stationarity / regime assumptions
  - Conditional effect function \(\tau(x)\) is stable enough for the scoring population; otherwise abstain or restrict.
- Independence assumptions
  - Effects are per-customer and ignore interference; portfolio-level feedback handled downstream.
- Any “only valid when …” conditions
  - Only valid within overlap-supported regions; abstain/revert to control if propensities are extreme.

7) Mathematical definition:
- Output_1: \(\widehat{\tau}_{1,0}(x)\)
- Output_2: \(\widehat{\tau}_{2,0}(x)\)
- Output_3: \(\widehat{\pi}(x)=\arg\min_{a\in\{0,1,2\}} \widehat{\mathbb{E}}[Y(a)\mid X=x]\) (risk-only Step‑1), with \(\widehat{\mathbb{E}}[Y(a)\mid X=x]=\widehat{\mu}_0(x)+\widehat{\tau}_{a,0}(x)\) for \(a\neq 0\)

8) Validation & acceptance tests (must-pass):
- Test_1: Range and sign sanity — all CATEs finite; fraction of |CATE|>0.5 must be below a threshold (to flag extrapolation/instability).
- Test_2: Overlap gating — recommended_action must be 0 (control) for rows failing overlap/trimming rules; pass if gating is enforced 100% of the time.

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: output uncertainty intervals; include policy value estimates and abstain_reason codes


INTERFACE CONTRACT (v1)

1) Contract name:
- OfflinePolicyValueReport_v1

2) Owner + consumer(s):
- Owner agent: C
- Consuming agents: E, Lead Integrator, app/dashboard

3) Purpose (1-2 sentences):
- Provide an offline evaluation report of candidate policies using IPS/DR estimators, including sensitivity to trimming and propensity specification, so strategy modules and simulations can rely on a validated value signal.

4) Inputs (schema + units + timing):
- Input_1: name=logged_data, type=DataFrame, unit=n/a, frequency=per_training_batch, time_index=t, allowable_range=n/a, missingness_policy=must_include_T_Y_X
- Input_2: name=candidate_policies, type=List[policy_id], unit=n/a, frequency=per_run, time_index=t, allowable_range=n/a, missingness_policy=required
- Input_3: name=propensity_specs, type=List[spec_id], unit=n/a, frequency=per_run, time_index=t, allowable_range=n/a, missingness_policy=required
- Input_4: name=trimming_rules, type=List[rule_id], unit=n/a, frequency=per_run, time_index=t, allowable_range=n/a, missingness_policy=required

5) Outputs (schema + units + interpretation):
- Output_1: name=policy_value_ips, type=Dict[policy_id,float], unit=expected_default_rate, allowable_range=[0,1], interpretation=estimated \(V(\pi)\) via IPS (lower is better)
- Output_2: name=policy_value_dr, type=Dict[policy_id,float], unit=expected_default_rate, allowable_range=[0,1], interpretation=estimated \(V(\pi)\) via DR (lower is better)
- Output_3: name=sensitivity_summary, type=Dict, unit=n/a, allowable_range=n/a, interpretation=stability metrics across propensity_specs and trimming_rules

6) Assumptions:
- Identification/causality assumptions (if any)
  - Same as causal engine: SUTVA, ignorability given X, overlap within trimmed set.
- Stationarity / regime assumptions
  - Logged policy and evaluation population are comparable after conditioning/trimming.
- Independence assumptions
  - Observations are treated as independent for value estimation; clustering handled later if needed.
- Any “only valid when …” conditions
  - Only valid when overlap diagnostics pass; otherwise values are reported as unreliable.

7) Mathematical definition:
- IPS value: \(\widehat{V}_{IPS}(\pi)=\frac{1}{n}\sum_{i=1}^n \frac{\mathbf{1}\{T_i=\pi(X_i)\}Y_i}{\hat e_{T_i}(X_i)}\)
- DR value (schematic): combines IPS with outcome regression \(\hat\mu_t(x)\) to reduce variance and add robustness.

8) Validation & acceptance tests (must-pass):
- Test_1: Placebo policy — include a random policy baseline; pass if it does not outperform meaningful candidates under DR.
- Test_2: Trimming sensitivity — policy ordering should be stable across reasonable trimming rules; if unstable, report “do not deploy” gate.

9) Versioning & change control:
- Version: v1
- Breaking change rule: any change to units/ranges/schema requires Lead Integrator approval
- Proposed future extensions: add profit-based utility and constraints; add confidence intervals via bootstrap