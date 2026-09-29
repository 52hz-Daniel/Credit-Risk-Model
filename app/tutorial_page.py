import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_tutorial_page():
    """
    Renders the Undergrad STEM Tutorial Page:
    Bridging intuition, mathematics, and code from first principles.
    Designed for 1st/2nd year undergrad STEM students with an engaging tutor tone.
    """
    # Header & Tutor Introduction
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E293B, #0F172A); padding: 30px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
        <span style="background-color: #38BDF8; color: #0F172A; font-weight: 700; padding: 4px 12px; border-radius: 20px; font-size: 13px;">STEM UNDERGRAD TUTORIAL</span>
        <h1 style="color: #F8FAFC; margin-top: 12px; margin-bottom: 8px; font-size: 32px;">The Complete Journey: From Intuition to Institutional AI Credit Strategy</h1>
        <p style="color: #94A3B8; font-size: 16px; line-height: 1.6; margin-bottom: 0px;">
            Hey there! Welcome to the math & quantitative lab. If you're a first or second-year STEM student (Computer Science, Math, Stats, Physics, or Engineering), you already know basic calculus, intro probability, and maybe some Python. But how do you go from classroom theory to solving real, multi-million-dollar financial decisions under uncertainty?
            <br><br>
            In this guide, we'll walk you through this entire project <b>step-by-step</b>. We won't just dump code on you. We will tell the <b>real story</b>: starting from the simplest intuitive question, discovering where basic models fail, and climbing the mathematical ladder until we have a battle-tested, regulation-compliant AI engine. Grab a coffee—let's dive in! ☕
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Table of Contents / Roadmap
    with st.expander("🗺️ Guided Tour: What We Will Learn Today", expanded=False):
        st.markdown("""
        1. **The Starting Dilemma**: Capital Markets vs. Retail Credit (The Volume Problem)
        2. **Chapter 1: The Macro Weather** — Hamilton (1989) Markov-Switching Regimes
        3. **Chapter 2: The Chaos Measurement** — Claude Shannon & Information Entropy
        4. **Chapter 3: The Tree Whisperer** — XGBoost & The Danger of Overconfidence (Platt Scaling)
        5. **Chapter 4: The Black Box & The Law** — OSFI Guideline E-23 & Scott Lundberg's SHAP
        6. **Chapter 5: The Causal Revolution** — Moving from Prediction to Prescription (X-Learner & CATE)
        7. **Chapter 6: The Competitor's Shadow** — Retail Game Theory & The Poaching Dilemma
        8. **Chapter 7: The Virtual Society** — 24-Month Agent-Based Modeling (Mesa ABM)
        9. **Curated Resources**: YouTube Lessons, Textbooks, and Research Papers
        """)

    # --------------------------------------------------------------------------
    # PROLOGUE
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🏛️ Prologue: The Volume Dilemma")
    
    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        st.write("""
        Imagine you're the Chief Risk Officer at a major bank like RBC or TD. 
        
        If you work in **Capital Markets**, you might lend **$500 Million** to an airline to buy ten Boeing 787s. You have a dedicated team of 15 senior financial analysts who spend 4 months inspecting flight logs, fuel price hedges, and corporate balance sheets. If the airline defaults, it's a catastrophe.
        
        Now step into **Retail Credit Cards**. You aren't dealing with 1 big borrower. You have **3,000,000 everyday citizens**, each asking for a **$5,000 credit limit**.
        - You *cannot* hire an analyst for each applicant—the margins are far too thin.
        - Your system must approve or reject an application in **200 milliseconds** when someone taps 'Apply' on their phone.
        - If you are too cautious and reject too many people, you lose interest revenue and customer loyalty to rival banks.
        - If you are too reckless, defaults will wipe out your capital.
        
        So the central challenge of retail credit is **a volume and probability optimization game**. Let's build the mathematics to solve it.
        """)
    with col_p2:
        st.info("""
        **Key Takeaway**:
        Retail credit risk is not about preventing *every* default. It is about mathematically maximizing the **risk-adjusted net margin**:
        $$\\text{Net Margin} = \\text{Interest Revenue} - \\text{Expected Losses} - \\text{Operating Costs}$$
        """)

    # --------------------------------------------------------------------------
    # CHAPTER 1: MACRO REGIMES (HAMILTON MS-AR)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🌦️ Chapter 1: The Macro Weather — When Good People Face Bad Times")
    
    st.write("""
    ### The Intuitive Thought
    When most beginners build a credit scoring model, they look only at the person: *What is their income? How old are they? Have they missed a bill?*
    
    **Here is the flaw**: Nobody defaults in a vacuum! An honest software engineer making $120,000/year might have a 0.5% chance of default in a booming economy. But if the central bank hikes interest rates from 0.25% to 5.00%, tech startups lay off thousands, and mortgage payments double, that same person's default risk jumps to 15%.
    
    Traditional statistical models assume that the economic relationship between variables is **static (stationary)**:
    $$y_t = \\beta_0 + \\beta_1 x_t + \\epsilon_t$$
    *In real life, this is like assuming the road conditions in Toronto are always sunny.* But the real world has **seasons**: sunny summer highways vs. black ice blizzards.
    """)

    st.markdown("### The Mathematical Breakthrough: Hamilton (1989) Markov-Switching")
    st.write("""
    In 1989, quantitative economist **James D. Hamilton** published a revolutionary paper in *Econometrica*: 
    **["A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"](file:///Users/xiaoda/Cursor/Credit%20Risk/Research%20Reference/A%20New%20Approach%20to%20the%20Economic%20Analysis%20of%20Nonstationary%20Time%20Series%20and%20the%20%20Business%20Cycle.pdf)**.
    
    Hamilton proposed that the economy is governed by an **unobserved hidden regime variable** $S_t \\in \\{0, 1\\}$:
    - **Regime 0 (Calm / Expansion)**: Low credit spreads, steady employment, lower volatility.
    - **Regime 1 (Stress / Contraction)**: High credit spreads, elevated inflation/rates, rising default rates.
    
    The equation for the macro indicator $y_t$ (e.g. US High Yield Credit Spread from FRED) switches regimes:
    $$y_t = \\mu_{S_t} + \\sum_{j=1}^p \\phi_j (y_{t-j} - \\mu_{S_{t-j}}) + \\epsilon_t, \\quad \\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)$$
    
    The transition between calm and stress is governed by a **Markov Transition Matrix** $\\mathbf{P}$:
    $$\\mathbf{P} = \\begin{bmatrix} p_{00} & p_{01} \\\\ p_{10} & p_{11} \\end{bmatrix} = \\begin{bmatrix} \\mathbb{P}(S_t=0 \\mid S_{t-1}=0) & \\mathbb{P}(S_t=1 \\mid S_{t-1}=0) \\\\ \\mathbb{P}(S_t=0 \\mid S_{t-1}=1) & \\mathbb{P}(S_t=1 \\mid S_{t-1}=1) \\end{bmatrix}$$
    """)

    # Interactive Transition Diagram / Demo
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        ```mermaid
        stateDiagram-v2
            direction LR
            State0: State 0 (Calm Economy)
            State1: State 1 (Systemic Stress)
            State0 --> State0: p00 = 92% (Persistent Growth)
            State0 --> State1: p01 = 8% (Shock / Rate Hike)
            State1 --> State1: p11 = 85% (Prolonged Recession)
            State1 --> State0: p10 = 15% (Recovery)
        ```
        """)
    with c2:
        st.info("""
        **No Look-Ahead Bias Safeguard**:
        In regulatory risk, you cannot use *smoothed* probabilities (which use future data $t+1, \\dots, T$). 
        You must strictly use **filtered marginal probabilities**:
        $$\\text{regime\\_prob\\_stress}_t = \\mathbb{P}(S_t=1 \\mid \\mathcal{F}_t)$$
        conditioned *only* on past and current macro data $\\mathcal{F}_t$.
        """)

    st.markdown("""
    💡 **Further Learning Links**:
    - 📖 [Statsmodels Markov Regression Guide](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_regression.html) (The exact library we used)
    - 🎥 [Khan Academy: Introduction to Markov Chains](https://www.khanacademy.org/math/linear-algebra)
    - 📄 [Hamilton (1990) Analysis of Time Series Subject to Changes in Regime](file:///Users/xiaoda/Cursor/Credit%20Risk/Research%20Reference/ANALYSIS%20OF%20TIME%20SERIES%20SUBJECT%20TO%20%20%20CHANGES%20IN%20REGIME.pdf)
    """)

    # --------------------------------------------------------------------------
    # CHAPTER 2: SHANNON ENTROPY & INFORMATION GAIN
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🎲 Chapter 2: The Chaos Measurement — Claude Shannon & Information Theory")
    
    st.write("""
    ### The Intuitive Thought
    You have downloaded 30,000 real credit customer records. You have dozens of variables: Age, Sex, Marital Status, Education, Limit, Bill Amounts, Payment Amounts, Missed Payments.
    
    Which of these variables actually contain **pure predictive signal**, and which ones are just **useless noise**?
    
    In traditional data science, people often compute linear correlation ($r$). But correlation *only* measures straight lines! If a relationship is non-linear (e.g., risk is low for low balance, medium for medium balance, but explodes exponentially for high balance), correlation fails completely.
    """)

    st.markdown("### The Mathematical Breakthrough: Shannon Information Entropy (1948)")
    st.write("""
    In 1948, **Claude Shannon** (the father of the digital age) asked: *How do you measure the amount of uncertainty or 'surprise' in an event?*
    
    For a binary random variable $Y \\in \\{0: \\text{Pays}, 1: \\text{Defaults}\\}$ with default probability $p$:
    $$H(Y) = -p \\log_2(p) - (1-p) \\log_2(1-p) \\quad \\text{[units: bits]}$$
    """)

    # Interactive Entropy Curve Calculator
    st.subheader("🎛️ Interactive Mini-Lab: Play with Shannon Entropy")
    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        p_user = st.slider("Probability of Default (p)", min_value=0.01, max_value=0.99, value=0.22, step=0.01)
        h_val = - (p_user * np.log2(p_user) + (1 - p_user) * np.log2(1 - p_user))
        st.metric("Shannon Entropy H(Y)", f"{h_val:.4f} bits")
        st.caption("Notice how entropy peaks at p=0.50 (complete coin-flip chaos) and drops to 0 when an outcome is completely guaranteed!")
    with col_e2:
        p_curve = np.linspace(0.001, 0.999, 100)
        h_curve = - (p_curve * np.log2(p_curve) + (1 - p_curve) * np.log2(1 - p_curve))
        fig_ent = go.Figure()
        fig_ent.add_trace(go.Scatter(x=p_curve, y=h_curve, mode="lines", name="H(Y)", line=dict(color="#38BDF8", width=3)))
        fig_ent.add_trace(go.Scatter(x=[p_user], y=[h_val], mode="markers", name="Your Portfolio Point", marker=dict(color="#F43F5E", size=12)))
        fig_ent.update_layout(title="Shannon Entropy Curve H(p) in Bits", xaxis_title="Default Probability p", yaxis_title="Entropy (Bits)", template="plotly_dark", height=280)
        st.plotly_chart(fig_ent, use_container_width=True)

    st.markdown("### How We Used This: Information Gain (Feature Screening)")
    st.write("""
    If we know a borrower's attribute $X$ (e.g. recent missed payment status), how much does that **reduce our uncertainty** about whether they will default? That is **Information Gain (Mutual Information)**:
    $$IG(Y, X) = H(Y) - H(Y \\mid X) = H(Y) - \\sum_{x \\in \\mathcal{X}} p(x) H(Y \\mid X = x)$$
    
    When we ran this on our 30,000 real accounts in [`models/feature_engineering.py`](file:///Users/xiaoda/Cursor/Credit%20Risk/models/feature_engineering.py), we uncovered a massive real-world truth:
    """)

    ig_table = pd.DataFrame([
        {"Feature": "pay_recent_delinquency (Latest missed payment)", "Information Gain (Bits)": 0.1084, "Significance": "⭐⭐⭐⭐⭐ Top Predictor"},
        {"Feature": "pay_max_delinquency (Worst 6m delinquency)", "Information Gain (Bits)": 0.0965, "Significance": "⭐⭐⭐⭐⭐ Core Behavior"},
        {"Feature": "avg_pay_amt (Monthly repayment velocity)", "Information Gain (Bits)": 0.0228, "Significance": "⭐⭐⭐ Solvency Buffer"},
        {"Feature": "limit_bal (Assigned credit limit)", "Information Gain (Bits)": 0.0220, "Significance": "⭐⭐⭐ Exposure Size"},
        {"Feature": "pay_to_bill_ratio (Cash flow coverage)", "Information Gain (Bits)": 0.0183, "Significance": "⭐⭐⭐ Liquidity Health"},
        {"Feature": "education (College / High School)", "Information Gain (Bits)": 0.0042, "Significance": "⭐ Negligible Signal"},
        {"Feature": "age (Borrower age)", "Information Gain (Bits)": 0.0026, "Significance": "⭐ Barely Above Noise"},
        {"Feature": "marriage (Marital status)", "Information Gain (Bits)": 0.0007, "Significance": "❌ Pure Noise"}
    ])
    st.dataframe(ig_table, use_container_width=True)

    st.markdown("""
    💡 **Further Learning Links**:
    - 🎥 [3Blue1Brown: A Short Introduction to Entropy & Information Theory](https://www.youtube.com/watch?v=2s3aJfRr9gE)
    - 📖 [Claude Shannon (1948) The Mathematical Theory of Communication](https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication)
    """)

    # --------------------------------------------------------------------------
    # CHAPTER 3: XGBOOST & PROBABILITY CALIBRATION
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🌲 Chapter 3: The Tree Whisperer — XGBoost & The Danger of Overconfidence")
    
    st.write("""
    ### The Intuitive Thought
    How do computers make decisions? A single **Decision Tree** is just a flowchart:
    - *Is recent delinquency > 1 month?* $\\to$ Yes: High Risk, No: Check next condition.
    - *Is utilization > 80%?* $\\to$ Yes: High Risk, No: Safe.
    
    Where does a tree decide where to make a cut? **It cuts precisely where Information Gain is maximized!**
    
    A single decision tree, however, overfits easily. So in modern credit risk, we use **XGBoost (Extreme Gradient Boosting)**:
    - We build an ensemble of 150 trees.
    - Each tree does not start from scratch. Tree 2 is trained to predict the **residuals (errors)** of Tree 1. Tree 3 predicts the errors of Tree 2, and so forth.
    """)

    st.markdown("### The Hidden Trap: Why Raw AI Probabilities Lie (Calibration)")
    st.write("""
    In retail banking, only $\\approx 22\\%$ of customers default. If an algorithm simply guessed *"nobody defaults"*, it would be 78% accurate while losing millions of dollars!
    
    To fix this, we apply **cost-sensitive balancing**:
    $$\\text{scale\\_pos\\_weight} = \\frac{N_{\\text{neg}}}{N_{\\text{pos}}} = \\frac{18,691}{5,309} \\approx 3.52$$
    This forces the tree to care 3.52 times more about catching a defaulting borrower.
    
    **The unintended consequence:** The raw probabilities output by the tree are now pushed violently toward 1.0! A raw score of 0.85 does *not* mean 85% of borrowers will default. In financial engineering, this is catastrophic.
    
    ### The Mathematical Cure: Out-of-Fold Platt Scaling
    In 1999, John Platt formulated **Platt Scaling**: We train a sigmoid logistic mapper over the tree's margin log-odds $f(x)$ on validation holdouts:
    $$\\hat{p}_{\\text{calibrated}} = \\frac{1}{1 + \\exp(A \\cdot f(x) + B)}$$
    """)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.metric("Expected Calibration Error (ECE)", "0.0170 (1.7%)", "Passing OSFI E-23 Decile Test")
        st.caption("Across every single decile (from 0-10% risk to 90-100% risk), our predicted probability deviates from the real observed default frequency by less than 1.7 percentage points.")
    with col_c2:
        st.metric("Area Under ROC Curve (ROC-AUC)", "0.7800", "Strong Separation")
        st.caption("78% probability that a randomly chosen defaulter has a higher predicted risk score than a randomly chosen non-defaulter.")

    st.markdown("""
    💡 **Further Learning Links**:
    - 🎥 [StatQuest: XGBoost Part 1 (Regression & Classification Intuition)](https://www.youtube.com/watch?v=OtD8wVxBQgE)
    - 📖 [Scikit-Learn Guide on Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
    - 📄 [Hastie, Tibshirani, Friedman: The Elements of Statistical Learning](file:///Users/xiaoda/Cursor/Credit%20Risk/Research%20Reference/The%20Elements%20of%20Statistical%20Learning.pdf)
    """)

    # --------------------------------------------------------------------------
    # CHAPTER 4: SHAP EXPLAINABILITY (COOPERATIVE GAME THEORY)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🔍 Chapter 4: The Black Box & The Law — Scott Lundberg & SHAP")
    
    st.write("""
    ### The Legal Mandate (OSFI Guideline E-23 & Adverse Action)
    Imagine a young professional applies for a credit card. The XGBoost model calculates a 42% default risk and rejects them.
    
    Under Canadian banking regulation (**OSFI Guideline E-23**) and US Federal Law (**Equal Credit Opportunity Act**), a bank **cannot** say:
    > *"Sorry, our neural network with 10 million parameters said no, but we don't know why."*
    
    The bank is legally required to send an **Adverse Action Notice** citing the exact top 4 reasons for rejection. How do you extract exact, fair reasons out of a non-linear ensemble of 150 decision trees?
    """)

    st.markdown("### The Mathematical Solution: Lloyd Shapley's 1953 Nobel Prize Formula")
    st.write("""
    In 1953, mathematician **Lloyd Shapley** solved how to fairly divide the winnings of a coalition of players in a cooperative game. In 2017, **Scott Lundberg** applied this to machine learning in the landmark paper:
    **["A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)](file:///Users/xiaoda/Cursor/Credit%20Risk/Research%20Reference/A%20Unified%20Approach%20to%20Interpreting%20Model%20%20Predictions.pdf)**.
    
    Each feature (income, credit limit, missed payment) is treated as a **player in a game**, and the final predicted default probability is the **payout**:
    $$\\phi_i(x) = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F| - |S| - 1)!}{|F|!} \\left[ f(S \\cup \\{i\\}) - f(S) \\right]$$
    
    **The 3 Axioms of Fairness:**
    1. **Efficiency**: The sum of all SHAP values equals the difference between the model's prediction and the base population rate: $\\sum \\phi_i = f(x) - \\mathbb{E}[f(X)]$.
    2. **Symmetry**: If two features contribute equally across all subsets, their SHAP values are identical.
    3. **Dummy / Null**: A feature with no impact receives exactly zero attribution ($\phi_i = 0$).
    """)

    st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png", caption="TreeSHAP: Decomposing tree ensemble output into additive Shapley values")

    st.markdown("""
    💡 **Further Learning Links**:
    - 🎥 [Scott Lundberg: Explaining Machine Learning Predictions with SHAP (NeurIPS Talk)](https://www.youtube.com/watch?v=B-c8tIgchu0)
    - 💻 [Official slundberg/shap GitHub Repository](https://github.com/slundberg/shap)
    """)

    # --------------------------------------------------------------------------
    # CHAPTER 5: CAUSAL INFERENCE (X-LEARNER)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🎯 Chapter 5: The Causal Revolution — From Prediction to Prescription")
    
    st.write("""
    ### The Doctor's Dilemma
    This is the most critical conceptual leap in modern artificial intelligence:
    
    > **Prediction is NOT Strategy.**
    
    Suppose an AI predicts that a patient with pneumonia has an 80% mortality risk. That is a **prediction**.
    Does that tell the doctor *what medication to prescribe*? No! If the doctor gives the wrong drug, the patient dies faster.
    
    In banking:
    - XGBoost tells you: *"Customer #120 has a 35% chance of default."*
    - The traditional bank manager panics and **cuts their credit limit by 50%**.
    - What happens? The customer needed that credit line to pay rent while waiting for their next paycheck. By cutting their limit, the bank **forced them into immediate bankruptcy**!
    """)

    st.markdown("### The Fundamental Problem of Causal Inference")
    st.write("""
    For any borrower, there exist multiple **Potential Outcomes** (Neyman-Rubin Causal Model):
    - $Y_i(0)$: Will they default if we **Do Nothing**?
    - $Y_i(1)$: Will they default if we **Cut their Limit by 20%**?
    - $Y_i(2)$: Will they default if we give them a **3-Month Payment Holiday**?
    
    You can only apply ONE action in real life. You **never observe the counterfactual**!
    We need to estimate the **Conditional Average Treatment Effect (CATE)**:
    $$\\tau_{a,0}(x) = \\mathbb{E}[Y(a) - Y(0) \\mid X = x]$$
    """)

    st.markdown("### The Algorithm: Multi-Arm X-Learner (Künzel et al. PNAS 2019)")
    st.write("""
    In [`models/causal_engine.py`](file:///Users/xiaoda/Cursor/Credit%20Risk/models/causal_engine.py), we implemented the **X-Learner** from the prestigious paper:
    **["Metalearners for estimating heterogeneous treatment effects using machine learning"](file:///Users/xiaoda/Cursor/Credit%20Risk/Research%20Reference/Metalearners%20for%20estimating%20heterogeneous%20treatment%20%20effects%20using%20machine%20learning.pdf)**.
    """)

    st.markdown("""
    ```mermaid
    flowchart TD
        Stage1["Stage 1: Base Outcome Models<br/>Train mu_0(x) on Control and mu_1(x) on Treated"]
        Stage2["Stage 2: Counterfactual Imputation<br/>D_1 = Y_1 - mu_0(X_1) and D_0 = mu_1(X_0) - Y_0"]
        Stage3["Stage 3: Second-Stage Effect Regressors<br/>Train tau_1(x) on Treated and tau_0(x) on Control"]
        Stage4["Stage 4: Propensity Weighting<br/>tau(x) = e(x)*tau_0(x) + (1 - e(x))*tau_1(x)"]
        Stage1 --> Stage2 --> Stage3 --> Stage4
    ```
    """)

    st.success("""
    **Real Discovery from Our Dataset**:
    - For borrowers undergoing severe liquidity distress, **Payment Holiday (T=2)** drops default probability by **7.4%** by providing a cash runway.
    - For borrowers with high discretionary spending, **Limit Cut (T=1)** saves money without pushing them over the edge.
    """)

    st.markdown("""
    💡 **Further Learning Links**:
    - 📖 [Judea Pearl: The Book of Why — The New Science of Cause and Effect](https://en.wikipedia.org/wiki/The_Book_of_Why)
    - 💻 [Uber CausalML Library Reference](https://github.com/uber/causalml)
    - 💻 [Microsoft EconML Library Reference](https://github.com/py-why/EconML)
    """)

    # --------------------------------------------------------------------------
    # CHAPTER 6: RETAIL GAME THEORY (POACHING)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## ♟️ Chapter 6: The Competitor's Shadow — Retail Game Theory")
    
    st.write("""
    ### The Blind Spot of Isolated Optimization
    If you only optimize default risk, you make foolish business mistakes.
    
    Suppose Customer #5 has a $50,000 credit limit, uses only 17% of it, pays in full every month, and has a 750 credit score.
    - Causal AI says: *"Cutting their limit to $40,000 drops default risk from 8.4% to 5.7%."*
    - But what does the customer do when they get an SMS saying: *"We have reduced your credit limit"*?
    - **They get furious, cancel their card, and switch to American Express or RBC!**
    
    This is **Competitor Poaching Risk**.
    """)

    st.markdown("### The Game-Theoretic Expected Profit Equation")
    st.write("""
    In [`models/retail_game_theory.py`](file:///Users/xiaoda/Cursor/Credit%20Risk/models/retail_game_theory.py), we model this as a non-cooperative game between our bank and rival lenders. We formulate the **Net Objective Function**:
    $$\\pi^*(x) = \\arg\\max_{t \\in \\{0, 1, 2\\}} \\left[ \\underbrace{\\text{Net Interest Margin}(t)}_{\\text{Revenue}} - \\underbrace{\\hat{P}(\\text{default} \\mid t) \\cdot \\text{EAD} \\cdot \\text{LGD}}_{\\text{Expected Default Loss}} - \\underbrace{P_{\\text{poach}}(t) \\cdot \\text{Customer LTV}}_{\\text{Attrition Penalty}} \\right]$$
    
    When we include Customer Lifetime Value ($LTV$), the optimal strategy for prime customers flips back to **Do Nothing (Control)**, preserving over **$700 in net profit** per customer that would otherwise be destroyed by naive risk cuts!
    """)

    st.markdown("""
    💡 **Further Learning Links**:
    - 🎥 [Yale University Game Theory Course (Prof. Ben Polak)](https://www.youtube.com/watch?v=nM3rTU9ci88)
    - 📖 [Avinash Dixit & Barry Nalebuff: Thinking Strategically](https://en.wikipedia.org/wiki/Thinking_Strategically)
    """)

    # --------------------------------------------------------------------------
    # CHAPTER 7: AGENT-BASED MODELING (ABM)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🌐 Chapter 7: The Virtual Society — 24-Month Agent-Based Simulation (Mesa)")
    
    st.write("""
    ### Why Static Equations Aren't Enough
    In economics, closed-form equations fail when feedback loops exist. When thousands of people lose jobs, bankruptcies cascade, banks tighten credit, which shrinks the economy further.
    
    To prove that our Causal AI strategy works over time, we built a **digital twin of the economy** using the **Mesa 3.5 Agent-Based Modeling framework** in [`simulation/abm_engine.py`](file:///Users/xiaoda/Cursor/Credit%20Risk/simulation/abm_engine.py):
    - **1,000 autonomous `CustomerAgents`** receiving paychecks, paying expenses, paying credit card balances, and facing macroeconomic shocks.
    - A **`BankAgent`** choosing between:
      1. *Traditional Strategy*: Blanket 50% limit cuts whenever a bill is missed.
      2. *Causal AI Strategy*: Targeted forbearance (holidays) + targeted limit cuts + poaching defense.
    - An economic clock running for **24 simulated months**, driven by real Bank of Canada interest rate and spread cycles.
    """)

    # ABM Headline Results
    st.subheader("🏆 The Final 24-Month Scoreboard")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Defaults Averted", "51 Borrowers", "-25.2% Fewer Bankruptcies")
    with k2:
        st.metric("Total Profit Lift", "+$3,406,235", "+42.8% Higher Retained Margin")
    with k3:
        st.metric("Simulation Velocity", "1.8 Seconds", "Fully Vectorized Batch Execution")

    st.markdown("""
    💡 **Further Learning Links**:
    - 📖 [Project Mesa: Agent-based modeling in Python](https://mesa.readthedocs.io/latest/)
    - 📖 [Epstein & Axtell: Growing Artificial Societies (MIT Press)](https://mitpress.mit.edu/9780262550253/growing-artificial-societies/)
    - 🏛️ [Bank of Italy: BeforeIT.jl Institutional Macroeconomic Agent-Based Framework](file:///Users/xiaoda/Cursor/Credit%20Risk/Reference%20Git%20Repo/BeforeIT.jl)
    """)

    # --------------------------------------------------------------------------
    # EPILOGUE & CURATED RESOURCES
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📚 Epilogue: Your Roadmap to Mastery")
    st.write("""
    If you've read this far, congratulations! You now understand the full intellectual continuum of modern institutional quantitative finance:
    $$\\text{Real Macro Time Series} \\longrightarrow \\text{Information Entropy} \\longrightarrow \\text{Calibrated ML} \\longrightarrow \\text{Game Theory XAI} \\longrightarrow \\text{Causal Uplift} \\longrightarrow \\text{ABM Simulation}$$
    
    Here is a curated reading list to take your skills to the absolute top 1% of STEM undergraduates:
    """)

    st.markdown("""
    ### 📺 Essential YouTube Channels
    1. **[StatQuest with Josh Starmer](https://www.youtube.com/@statquest)** — The clearest visual intuition for Decision Trees, XGBoost, and ROC curves anywhere on the internet.
    2. **[3Blue1Brown (Grant Sanderson)](https://www.youtube.com/@3blue1brown)** — Watch the Essence of Linear Algebra, Neural Networks, and Information Theory.
    3. **[MIT OpenCourseWare (18.06 Linear Algebra - Gilbert Strang)](https://www.youtube.com/playlist?list=PLE7DDD91010BC51F8)** — The bedrock of all machine learning mathematics.

    ### 📖 Definitive Textbooks
    1. **The Elements of Statistical Learning** (*Hastie, Tibshirani, Friedman*) — The undisputed bible of machine learning.
    2. **Time Series Analysis** (*James D. Hamilton*) — The definitive text on Markov switching models.
    3. **Causal Inference: What If** (*Hernán & Robins, Harvard University*) — Free online textbook on potential outcomes and causal inference.
    4. **Elements of Information Theory** (*Cover & Thomas*) — The fundamental physics of data and entropy.

    ### 💻 Open Source Repositories to Star on GitHub
    - [GitHub: 52hz-Daniel/Credit-Risk-Model](https://github.com/52hz-Daniel/Credit-Risk-Model) — This repository!
    - [GitHub: slundberg/shap](https://github.com/slundberg/shap) — SHAP TreeExplainer source code.
    - [GitHub: uber/causalml](https://github.com/uber/causalml) — Uplift modeling & X-Learner implementations.
    - [GitHub: projectmesa/mesa](https://github.com/projectmesa/mesa) — Agent-Based Modeling framework.
    """)
