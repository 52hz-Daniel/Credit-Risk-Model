# Can a credit-risk model help a lender make better decisions?

When a credit-card customer starts to struggle, a lender has several choices: leave the account as it is, lower the credit limit, or offer temporary payment relief. Each choice may affect both the customer and the lender. **This project asks whether we can estimate the customer's payment risk, understand the reasons for that estimate, and test possible responses before anyone uses them in practice.**

EQUITY-TWIN is an educational prototype. It combines a historical credit-card dataset, economic indicators, machine-learning models, and a simulated lending portfolio. It does **not** make real lending decisions or show that a bank has prevented defaults or earned additional profit.

## The project in one minute

1. **Predict payment trouble.** Use an account's recent bills, payments, and repayment history to estimate the chance of a missed payment.
2. **Check whether the estimate is useful.** Compare it with what actually happened in a separate test set of 6,000 accounts.
3. **Explore a tougher economy.** Change economic inputs, such as interest rates or credit-market stress, and see how the model's predicted risks move. This is a scenario, not a forecast.
4. **Explain a prediction.** Show which inputs pushed a particular model score up or down.
5. **Compare possible actions.** Estimate what might happen under no change, a smaller credit limit, or a payment holiday, then run those rules through a 24-month computer simulation.

The central distinction is simple: **predicting who may miss a payment is different from knowing which action will help them.** Historical repayment data can be used to test predictions. Claims about how an unobserved action would change an outcome need separate evidence and assumptions.

## What data are we using?

The credit-card table has **30,000 accounts**; **6,000** are reserved for the displayed test comparison. About **22.12%** of the accounts have the dataset's default-payment label. The account fields and record count match the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), which describes credit-card customers in Taiwan in 2005. These are historical records, not a current Canadian bank portfolio. The dataset's outcome is a default-payment label; do not interpret it as a verified 90-day bankruptcy or write-off outcome.

The app also displays economic series from the Bank of Canada and the US Federal Reserve's FRED database. Those newer series do not give us observed outcomes for the same borrowers in a later Canadian recession. Connecting an economic scenario to these historical accounts is therefore a modeling exercise.

## What do the prediction results say?

The app compares a model using customer information with a **macro-aware** version that also uses economic inputs. On the displayed holdout set:

| Measure | Macro-aware model | Compared with customer-only model | Plain-English meaning |
| --- | ---: | ---: | --- |
| ROC-AUC | 0.7792 | 0.0008 lower | How well the model ranks accounts that later miss payments above accounts that do not. |
| PR-AUC | 0.5574 | 0.0015 lower | How well it identifies missed payments when the two outcomes have different frequencies. |
| Brier score | 0.1356 | 0.0001 higher | Average error in predicted probabilities; lower is better. |
| Expected calibration error | About 1.70% | 0.03 percentage points lower | Average gap between predicted risk and observed frequency across groups of similar scores; lower is better. |

**What we learned:** the macro-aware model does **not** improve the displayed holdout ranking or Brier score. Its calibration difference is small. This comparison does not establish that economic inputs improve prediction for these accounts. The stress controls are still useful for exploring assumptions about changing conditions.

Calibration deserves a concrete example: if a group of 100 accounts is each assigned roughly 20% risk, we expect about 20 missed payments in that group. We can compare that expectation with the observed count. The app's calibration error summarizes such gaps across groups; it is **not** “98.3% accuracy,” a guarantee for an individual customer, or an OSFI pass mark.

## Why are these methods here?

| Term on the app | What it means here | Why it matters |
| --- | --- | --- |
| **XGBoost** | Many small decision trees are combined to estimate payment risk. | It can learn patterns that a single simple rule might miss. |
| **Platt scaling / calibration** | A second model adjusts raw scores so a stated probability better matches observed frequencies on validation data. | A risk score should mean something measurable, not just rank people. |
| **Markov switching** | A time-series model allows economic data to move between states such as calmer and more stressed periods. The chance of next month's state depends on the current state. | It creates alternative economic conditions for stress tests. The states are estimated, not official recession labels. |
| **Dynamic regression** | A model relates an outcome to changing inputs over time, sometimes including earlier values of the outcome or inputs. | It asks how an economic shock might carry forward. An economic relationship by itself does not prove what will happen to a particular borrower. |
| **Entropy / information gain** | A way to measure uncertainty and how much a variable reduces it. | It helps explore which account fields contain information about payment outcomes. A useful predictor is not necessarily a cause. |
| **SHAP** | A method that breaks down a model's score into the contributions of its inputs, relative to a reference score. | It helps inspect why the model gave a particular score. A SHAP contribution describes the model; it does not show that changing that input would change the customer's outcome. |
| **Causal intervention / X-learner** | A method for estimating how different actions might change an outcome for different people. | It frames the decision question: would a limit change or payment holiday help? Credible causal estimates require suitable treatment-outcome data and defensible assumptions. |
| **Agent-based simulation (ABM)** | A computer model in which simulated customers and a simulated lender interact over repeated months. | It tests how decision rules behave *inside the chosen simulation*. Its results depend on the rules and assumptions programmed into it. |
| **OSFI E-23** | Canadian guidance on how federally regulated institutions should govern, document, review, and monitor models. | It is a useful reference for model-risk questions. Citing it or displaying SHAP does not make this prototype compliant. |

## What happens when we compare lending actions?

For a selected account, the app displays estimated outcomes under **no change**, a **20% limit cut**, and a **three-month payment holiday**. The difference between an action and no change is called a *treatment effect*. A negative estimated effect on missed-payment risk would favor that action in the model.

But we cannot observe all three outcomes for the same person in real life. The historical credit-card table records repayment behavior, not a controlled comparison of these three policies. Treat the action rankings as **exploratory model outputs**, not evidence that a specific customer should have a credit limit cut or receive relief. The next step for a causal claim would be credible action and outcome data, a clear study design, and checks for who received each action and why.

The 24-month simulation compares policies for virtual customers. Figures such as “25.2% fewer defaults” and “$3.42 million more profit,” when shown, are **outputs of that simulation**, not observed bank results. They should be reported with the scenario, starting portfolio, assumptions, and uncertainty. The simulated profit is especially sensitive to the assumed losses, interest income, customer behavior, and cost of payment relief.

## What should a lender check before any real use?

- **Data fit:** Can a model built on 2005 Taiwanese credit-card records represent today's borrowers, products, and economy?
- **Prediction quality:** Does it beat a simple customer-only baseline on fresh, time-separated data? Are its probabilities still calibrated?
- **Action evidence:** Are limit changes and payment holidays evaluated with credible treatment and outcome data?
- **Fairness:** How do errors and decisions differ across groups, and could other inputs act as proxies? A single “parity” percentage cannot settle this.
- **Governance:** Can reviewers trace the data, assumptions, model versions, decisions, and monitoring results? OSFI's [Guideline E-23](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/guideline-e-23-model-risk-management-2027) is a reference for that work; institutional compliance requires much more than a dashboard.

**Takeaway:** The demonstrated result is a test-set comparison of payment-risk models. The economic scenarios, proposed actions, and portfolio outcomes are useful experiments, with stronger evidence needed before they can guide real lending.
