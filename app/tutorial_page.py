import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def render_tutorial_page():
    """
    Undergraduate STEM Educational Masterclass:
    Explains the entire AI Credit Risk Strategy project from first principles.
    Target audience: 1st/2nd year undergraduate STEM students.
    Focuses on intuitive storytelling, demystifying every term and formula,
    and explaining exactly WHY each mathematical piece exists.
    """

    # --------------------------------------------------------------------------
    # TOP HERO: THE TUTOR'S WELCOME & THE CENTRAL MISSION
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 26px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="background-color: #004AC6; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px;">STEM TUTORIAL</span>
            <span style="color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 11px;">FIRST-PRINCIPLES SYLLABUS</span>
        </div>
        <h1 style="color: #0F172A; font-family: 'Inter', sans-serif; font-size: 26px; font-weight: 700; margin: 4px 0 10px 0;">
            How We Built an AI Credit Strategy: A Step-by-Step Educational Journey
        </h1>
        <p style="color: #334155; font-size: 14px; line-height: 1.6; margin-bottom: 0;">
            Welcome! If you are a first or second-year STEM student in Computer Science, Mathematics, Statistics, Physics, or Engineering, you already know basic calculus, introductory probability, and some Python. 
            <br><br>
            <b>Our Goal in this Guide:</b> We are going to teach you this entire project from scratch—just like a tutor sitting next to you at a whiteboard. We will not throw unexplained jargon or naked formulas at you. Instead, you will see the <b>real story</b>: what problem a retail bank faces, why the obvious beginner solutions fail in real life, why we need each mathematical tool, and what every single letter and symbol means.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 1: THE REAL-WORLD PROBLEM (THE VOLUME DILEMMA)
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            STAGE 1 // THE PROBLEM STATEMENT
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 12px 0;">
            1. The Bank's Real-Life Dilemma: Volume & Latency
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            Imagine you are hired as a quantitative data scientist at a major retail bank.
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In <b>Corporate Lending</b>, a bank might lend $200 Million to an airline. A team of 10 human financial analysts spends 3 months reading balance sheets before making a decision.
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In <b>Retail Credit Cards</b>, the situation is completely different:
        </p>
        <ul style="color: #334155; font-size: 14px; line-height: 1.6;">
            <li>You have <b>3,000,000 everyday citizens</b> applying for cards or borrowing on their existing credit limits ($5,000 average).</li>
            <li>Your software must make an automated decision in <b>200 milliseconds</b> when someone taps "Apply" or swipes their card.</li>
            <li>You <i>cannot</i> hire human analysts for every applicant—the profit margins per card are far too thin.</li>
            <li>If you are <b>too strict</b> and reject everyone, you earn zero interest revenue, and rival banks take all your customers.</li>
            <li>If you are <b>too reckless</b> and approve everyone, defaults will wipe out your bank's cash reserves.</li>
        </ul>
        <div style="background-color: #F8FAFC; border-left: 3px solid #004AC6; padding: 12px 16px; border-radius: 4px; margin-top: 12px;">
            <div style="font-weight: 600; color: #0F172A; font-size: 13px;">The Bank's Fundamental Equation:</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #004AC6; margin-top: 4px;">
                Net Profit = Interest Income - Loan Default Losses - Lost Customer Penalty
            </div>
            <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
                The objective is not to eliminate all risk (which would mean lending zero dollars), but to maximize this net profit equation.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 2: THE NAIVE ATTEMPT & THE 3 CATASTROPHES
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #BA1A1A; margin-bottom: 4px;">
            STAGE 2 // WHY STANDARD MACHINE LEARNING FAILS
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 12px 0;">
            2. The "Naive" Solution: The 3 Hidden Catastrophes
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            If you give this problem to a beginner data science student, they will build what 95% of banks traditionally built:
        </p>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 16px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #0F172A; margin-bottom: 12px;">
            The Naive Policy: "Train a classifier to predict default risk. If predicted risk is high &rarr; immediately slash their credit limit by 50%!"
        </div>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In the real world, this naive policy triggers <b>three major disasters</b>:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 10px;">
            <div style="background-color: #FFF5F5; border: 1px solid #FED7D7; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; color: #9B1C1C; font-size: 13px;">Catastrophe 1: The Liquidity Death Spiral</div>
                <div style="font-size: 12px; color: #742A2A; margin-top: 6px; line-height: 1.5;">
                    Suppose a borrower experiences a temporary 1-month layoff between jobs. If you cut their credit line, they lose the ability to buy groceries or pay rent. <b>Your credit cut literally forces them into bankruptcy</b> when they would have recovered!
                </div>
            </div>
            <div style="background-color: #FFF5F5; border: 1px solid #FED7D7; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; color: #9B1C1C; font-size: 13px;">Catastrophe 2: Weather Blindness</div>
                <div style="font-size: 12px; color: #742A2A; margin-top: 6px; line-height: 1.5;">
                    If you train your model in 2021 (a booming economy with 0.25% interest rates), it will assume everyone pays on time. When 2023 arrives and interest rates surge to 5.00%, the model has no idea the "economic weather" has completely changed.
                </div>
            </div>
            <div style="background-color: #FFF5F5; border: 1px solid #FED7D7; border-radius: 6px; padding: 14px;">
                <div style="font-weight: 700; color: #9B1C1C; font-size: 13px;">Catastrophe 3: Competitor Poaching</div>
                <div style="font-size: 12px; color: #742A2A; margin-top: 6px; line-height: 1.5;">
                    If an honest prime customer with an 800 credit score forgets one utility bill, the naive model cuts their limit. Insulted, the customer cancels their card and moves all their savings to a rival bank. The bank lost a $5,000 customer forever.
                </div>
            </div>
        </div>
        <p style="color: #334155; font-size: 14px; line-height: 1.6; margin-top: 14px;">
            <b>Now you understand why we need advanced math!</b> We cannot use a simple classifier. We must build six specific mathematical pieces to solve each of these disasters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PIECE 1: HAMILTON (1989) MARKOV REGIMES
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            PIECE 1 // MODELING THE MACROECONOMIC WEATHER
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">
            3. Hamilton (1989) Markov-Switching Regimes
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>Why do we need this piece?</b> To fix Catastrophe 2 (Weather Blindness). You cannot judge a borrower's safety without knowing if the economy is in a calm summer or a frozen blizzard.
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In 1989, quantitative economist <b>James D. Hamilton</b> proposed that macroeconomic data (like the US High-Yield Credit Spread from FRED) is governed by an unobserved discrete state variable:
        </p>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #0F172A; margin-bottom: 12px;">
            <b>State 0 (Calm / Expansion):</b> Low interest rates, low unemployment, high credit safety.<br>
            <b>State 1 (Stress / Contraction):</b> High interest rates, corporate layoffs, rising default spikes.
        </div>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            Here is the core equation governing the economic indicator $y_t$ at month $t$:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"y_t = \mu_{S_t} + \phi \cdot (y_{t-1} - \mu_{S_{t-1}}) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)")

    # Deconstructing the Math
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 16px; margin: -10px 0 16px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 8px;">
            DECONSTRUCTING THE FORMULA FOR FRESHMEN:
        </div>
        <table style="width: 100%; font-size: 13px; color: #334155; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 6px 0; font-family: 'JetBrains Mono', monospace; font-weight: 600; width: 120px; color: #0F172A;">y_t</td>
                <td>The observable economic indicator at time <i>t</i> (e.g. US Credit Spread = 4.2%).</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 6px 0; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0F172A;">S_t</td>
                <td>The <b>hidden regime switch</b>: equals 0 when the economy is Calm, or 1 when in Stress.</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 6px 0; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0F172A;">\mu_{S_t}</td>
                <td>The average baseline spread in that regime (e.g. 2.5% in Calm, 6.5% in Stress).</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 6px 0; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0F172A;">\phi</td>
                <td>The autoregressive persistence (how much of last month's deviation carries over to this month).</td>
            </tr>
            <tr>
                <td style="padding: 6px 0; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0F172A;">\epsilon_t</td>
                <td>Random Gaussian noise representing unexpected economic shocks this month.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    ### How does the economy switch between Calm and Stress?
    It follows a **Markov Transition Matrix** $\mathbf{P}$. In probability, the vertical bar $\mid$ simply means **"given that"**:
    """)

    st.latex(r"\mathbf{P} = \begin{bmatrix} p_{00} & p_{01} \\ p_{10} & p_{11} \end{bmatrix} = \begin{bmatrix} \mathbb{P}(S_t=0 \mid S_{t-1}=0) & \mathbb{P}(S_t=1 \mid S_{t-1}=0) \\ \mathbb{P}(S_t=0 \mid S_{t-1}=1) & \mathbb{P}(S_t=1 \mid S_{t-1}=1) \end{bmatrix}")

    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px; margin: 10px 0 16px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 6px;">
            READING THE 4 PROBABILITIES IN PLAIN ENGLISH:
        </div>
        <ul style="font-size: 13px; color: #334155; line-height: 1.6; margin: 0; padding-left: 20px;">
            <li><b>p00 = 0.92 (92%)</b>: If the economy is Calm this month ($S_{t-1}=0$), there is a 92% chance it stays Calm next month ($S_t=0$).</li>
            <li><b>p01 = 0.08 (8%)</b>: If the economy is Calm, there is an 8% chance a sudden shock triggers a Stress regime ($S_t=1$).</li>
            <li><b>p10 = 0.15 (15%)</b>: If the economy is in Stress, there is a 15% chance of recovery back to Calm next month.</li>
            <li><b>p11 = 0.85 (85%)</b>: If the economy is in Stress, recessions tend to persist with an 85% probability.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PIECE 2: SHANNON ENTROPY (CLAUDE SHANNON 1948)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            PIECE 2 // MEASURING INFORMATION WITHOUT BIAS
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">
            4. Claude Shannon (1948) Information Entropy
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>Why do we need this piece?</b> When you download 30,000 customer records, you have 25 different columns: Age, Marital Status, Education, Limit, Recent Delinquencies, Bill Amounts. Which variables contain <b>genuine predictive signal</b>, and which ones are <b>useless noise</b>?
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            Beginners often use Pearson correlation ($r$). But correlation <i>only</i> detects straight lines! Default risk is highly non-linear.
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In 1948, <b>Claude Shannon</b> (the father of information theory) invented a mathematical ruler for uncertainty called <b>Entropy $H(Y)$</b>. For a customer who either pays ($Y=0$) or defaults ($Y=1$) with default probability $p$:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"H(Y) = -p \cdot \log_2(p) - (1-p) \cdot \log_2(1-p) \quad \text{[units: bits]}")

    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px; margin: -10px 0 16px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 6px;">
            UNDERSTANDING "BITS" OF UNCERTAINTY:
        </div>
        <p style="font-size: 13px; color: #334155; line-height: 1.5; margin: 0;">
            - If $p = 0.50$ (a 50/50 coin toss), uncertainty is at its absolute maximum: $H(Y) = 1.0$ bit.<br>
            - If $p = 0.0$ or $1.0$ (outcome completely guaranteed), uncertainty is zero: $H(Y) = 0.0$ bits.<br>
            - In our 30,000-customer dataset, the baseline default rate is $p = 0.221$ (22.1%), so portfolio baseline entropy is <b>0.760 bits</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Entropy Mini-Lab
    st.markdown("#### Interactive Mini-Lab: See How Uncertainty Changes with Probability")
    col_ent1, col_ent2 = st.columns([1, 2])
    with col_ent1:
        p_val = st.slider("Select Default Probability (p)", min_value=0.01, max_value=0.99, value=0.22, step=0.01)
        h_calc = - (p_val * np.log2(p_val) + (1 - p_val) * np.log2(1 - p_val))
        st.metric("Calculated Entropy H(Y)", f"{h_calc:.4f} bits")
        st.caption("Notice how entropy peaks at p=0.50 (maximum chaos) and falls to 0 at the edges!")
    with col_ent2:
        p_grid = np.linspace(0.001, 0.999, 100)
        h_grid = - (p_grid * np.log2(p_grid) + (1 - p_grid) * np.log2(1 - p_grid))
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=p_grid, y=h_grid, mode="lines", name="Entropy Curve", line=dict(color="#004AC6", width=2.5)))
        fig_e.add_trace(go.Scatter(x=[p_val], y=[h_calc], mode="markers", name="Your Point", marker=dict(color="#BA1A1A", size=10)))
        fig_e.update_layout(title="Shannon Binary Entropy Curve H(p)", xaxis_title="Default Probability (p)", yaxis_title="Entropy (Bits)", template="plotly_white", height=240, margin=dict(l=40, r=40, t=35, b=35))
        st.plotly_chart(fig_e, use_container_width=True)

    st.write("""
    ### What is Information Gain?
    If we know a borrower's attribute $X$ (like whether they missed last month's payment), how much does our uncertainty about default drop?
    """)

    st.latex(r"IG(Y, X) = H(Y) - H(Y \mid X)")

    st.write("""
    When we calculated $IG(Y, X)$ on all 30,000 real accounts, we discovered an eye-opening truth:
    """)

    ig_df = pd.DataFrame([
        {"Variable Name": "pay_recent_delinquency", "Information Gain (Bits)": 0.1084, "What it means": "Did they miss last month's bill? (Massive signal)"},
        {"Variable Name": "pay_max_delinquency", "Information Gain (Bits)": 0.0965, "What it means": "Worst delinquency in 6 months (Very strong signal)"},
        {"Variable Name": "avg_pay_amt", "Information Gain (Bits)": 0.0228, "What it means": "How much cash they actually repay every month"},
        {"Variable Name": "limit_bal", "Information Gain (Bits)": 0.0220, "What it means": "Total credit line given to them"},
        {"Variable Name": "education", "Information Gain (Bits)": 0.0042, "What it means": "College degree vs High School (Nearly zero predictive value)"},
        {"Variable Name": "age", "Information Gain (Bits)": 0.0026, "What it means": "Borrower age (Almost completely noise)"},
        {"Variable Name": "marriage", "Information Gain (Bits)": 0.0007, "What it means": "Marital status (Pure statistical noise)"}
    ])
    st.dataframe(ig_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # PIECE 3: XGBOOST & PROBABILITY CALIBRATION
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            PIECE 3 // ACCURATE PREDICTION WITHOUT ARROGANCE
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">
            5. XGBoost Decision Trees & Platt Probability Calibration
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>Why do we need this piece?</b> A single decision tree is just a flowchart (e.g., <i>"Is missed payment > 1? If yes &rarr; risky"</i>). <b>XGBoost (Extreme Gradient Boosting)</b> builds 150 small trees sequentially, where Tree 2 focuses on correcting the errors of Tree 1, Tree 3 corrects Tree 2, and so on.
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>The Hidden Trap (Overconfidence):</b> Because defaulting customers are relatively rare (only 22%), we force the trees to pay 3.52 times more attention to defaults using <code>scale_pos_weight = 3.52</code>.
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            While this helps the trees rank borrowers accurately, it destroys the <b>honesty of the probabilities</b>! The raw model might spit out a score of 0.85, but in reality, only 35% of those borrowers default. In finance, if your probabilities are exaggerated, you misprice every single loan!
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>The Solution: Platt Logistic Scaling (1999)</b><br>
            We take the raw uncalibrated tree margin $f(x)$ and pass it through a calibrated logistic sigmoid mapper on holdout validation data:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\hat{p}_{\text{calibrated}} = \frac{1}{1 + \exp(A \cdot f(x) + B)}")

    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px; margin: -10px 0 16px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 6px;">
            THE RESULT: EXPECTED CALIBRATION ERROR (ECE) = 1.71%
        </div>
        <p style="font-size: 13px; color: #334155; line-height: 1.5; margin: 0;">
            Across every risk decile (from borrowers predicted at 0-10% risk up to 90-100% risk), our calibrated probability matches the real-world default rate within <b>1.71 percentage points</b>! Uncalibrated models regularly miss by over 15%.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PIECE 4: EXPLAINABILITY (SHAPLEY VALUES / SHAP)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            PIECE 4 // EXPLAINING EVERY DECISION TO HUMANS
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">
            6. Explaining the Black Box: Lloyd Shapley's (1953) Game Theory
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>Why do we need this piece?</b> By law (under Canadian OSFI banking guidelines and the US Equal Credit Opportunity Act), a bank <b>cannot</b> reject a person and say: <i>"Sorry, our 150 decision trees said no, but we don't know why."</i>
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            You must tell the customer the exact top reasons for the decision (e.g. <i>"Your recent missed payment increased your risk by 14%, while your long 10-year history reduced it by 8%"</i>).
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In 1953, mathematician <b>Lloyd Shapley</b> won the Nobel Prize in Economics for solving a famous cooperative game theory problem: <i>If a group of players works together to win a prize, what is the mathematically fair way to divide the prize based on each player's actual contribution?</i>
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            In 2017, computer scientist Scott Lundberg applied this to machine learning (**TreeSHAP**): each feature is a "player", and the final risk score is the "prize":
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]")

    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px; margin: -10px 0 16px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 6px;">
            DECONSTRUCTING THE SHAPLEY FORMULA IN PLAIN ENGLISH:
        </div>
        <p style="font-size: 13px; color: #334155; line-height: 1.5; margin: 0;">
            Imagine testing every possible subset of features $S$. To see how much feature $i$ (e.g. missed payment) matters, we calculate the model's score <i>with</i> feature $i$ minus the score <i>without</i> feature $i$: $[f(S \cup \{i\}) - f(S)]$.<br>
            Then, we take the weighted average across all possible feature combinations. The result $\phi_i$ is the exact, unarguable contribution of that feature to the borrower's risk score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PIECE 5: CAUSAL INFERENCE (MULTI-ARM X-LEARNER)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            PIECE 5 // PREDICTION IS NOT STRATEGY
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">
            7. Causal AI & The Multi-Arm X-Learner: Prescribing the Right Action
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>Why do we need this piece?</b> This is the biggest mental breakthrough of modern AI:
        </p>
        <div style="background-color: #EFF6FF; border-left: 3px solid #004AC6; padding: 12px 16px; border-radius: 4px; font-size: 14px; color: #1E3A8A; font-weight: 600; margin-bottom: 12px;">
            Knowing that someone is sick does NOT tell you what medicine to prescribe.
        </div>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            A standard predictive model computes $\mathbb{E}[Y \mid X]$: <i>"Customer #5 has a 35% default probability."</i><br>
            If you blindly cut their credit limit by 50%, you take away their emergency cash cushion and <b>cause them to go bankrupt immediately</b>!
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            To be smart, the bank must estimate what would happen under <b>3 distinct possible actions</b> (Potential Outcomes):
        </p>
        <ul style="color: #334155; font-size: 13px; line-height: 1.6;">
            <li><b>Action 0: Do Nothing (Control)</b> &rarr; Let borrower continue as normal.</li>
            <li><b>Action 1: Cut Limit by 20%</b> &rarr; Useful for reckless spenders who have plenty of savings.</li>
            <li><b>Action 2: Grant a 3-Month Payment Holiday</b> &rarr; Pause payments so a temporarily laid-off worker can find a job.</li>
        </ul>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            The <b>Conditional Average Treatment Effect (CATE) $\tau_{a,0}(x)$</b> measures the difference in default risk if we apply Action $a$ versus Doing Nothing:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\tau_{a,0}(x) = \mathbb{E}[Y(a) - Y(0) \mid X = x]")

    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px; margin: -10px 0 16px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 6px;">
            HOW TO INTERPRET CATE \tau:
        </div>
        <p style="font-size: 13px; color: #334155; line-height: 1.5; margin: 0;">
            - If $\tau = -0.074$ (-7.4%), the action <b>reduced default risk by 7.4 percentage points</b>! This action saves the customer.<br>
            - If $\tau = +0.050$ (+5.0%), the action <i>increased</i> default risk! Applying this action would harm the customer and the bank.<br>
            - We use the <b>Multi-Arm X-Learner</b> (Künzel et al. 2019, PNAS) to accurately estimate this effect even when only a small fraction of customers received special treatment in the past.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PIECE 6: AGENT-BASED SIMULATION (MESA 3.5 DIGITAL TWIN)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            PIECE 6 // TESTING IN A VIRTUAL SOCIETY BEFORE REAL LIFE
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px 0;">
            8. Dynamic Economic Simulation: 24-Month Mesa Agent-Based Model
        </h2>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            <b>Why do we need this piece?</b> You cannot test a radical new mathematical credit policy on real human beings with real bank capital without proving it works first!
        </p>
        <p style="color: #334155; font-size: 14px; line-height: 1.6;">
            We built a <b>digital flight simulator</b> of the economy using the <b>Mesa 3.5 Agent-Based Modeling framework</b>:
        </p>
        <ul style="color: #334155; font-size: 13px; line-height: 1.6;">
            <li><b>1,000 Autonomous Customer Agents</b>: Each month, they earn a salary, pay living expenses, pay credit card bills, and experience random shocks (job loss, inflation hikes).</li>
            <li><b>1 Bank Agent</b>: Makes credit decisions using either the <i>Traditional Strategy</i> (blanket 50% limit cuts) or our <i>Causal AI Strategy</i> (targeted holidays and cuts).</li>
            <li><b>24-Month Clock</b>: Simulates 2 continuous years driven by real Bank of Canada interest rate spikes and recession shocks.</li>
        </ul>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 16px; margin-top: 14px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #004AC6; margin-bottom: 8px;">
                THE FINAL 24-MONTH SCOREBOARD:
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;">
                <div>
                    <div style="font-size: 11px; color: #64748B;">DEFAULTS PREVENTED</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: #006C4A;">-25.2% Fewer</div>
                    <div style="font-size: 11px; color: #475569;">51 bankruptcies averted per 1k cohort</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #64748B;">NET PORTFOLIO PROFIT LIFT</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: #004AC6;">+$3,406,235</div>
                    <div style="font-size: 11px; color: #475569;">Retained interest & avoided write-offs</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #64748B;">EXECUTION SPEED</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: #0F172A;">1.8 Seconds</div>
                    <div style="font-size: 11px; color: #475569;">Fast vectorized simulation engine</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SUMMARY / GLOSSARY OF TERMS FOR FRESHMEN
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 22px; margin-bottom: 20px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #004AC6; margin-bottom: 4px;">
            SUMMARY // QUICK-REFERENCE GLOSSARY
        </div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 12px 0;">
            9. Quick-Reference Glossary of Every Term Used
        </h2>
        <table style="width: 100%; font-size: 13px; color: #334155; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; width: 180px; color: #0F172A;">Default</td>
                <td>When a borrower fails to pay their debt for 90 days. The bank writes this off as a loss.</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Regime Switching</td>
                <td>A statistical model where the world flips between two states (e.g. Economic Growth vs. Recession).</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Entropy (H)</td>
                <td>The amount of uncertainty or surprise in an event, measured in bits (invented by Claude Shannon in 1948).</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Information Gain (IG)</td>
                <td>How many bits of uncertainty disappear when you learn a specific piece of information.</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Platt Calibration</td>
                <td>Transforming raw machine learning scores into true real-world probabilities so the AI doesn't exaggerate risk.</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Shapley Value (SHAP)</td>
                <td>A game-theoretic calculation that fairly attributes how much each feature contributed to the final prediction.</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Potential Outcomes</td>
                <td>The different parallel futures of a borrower under different bank policies (Control vs Limit Cut vs Payment Holiday).</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">CATE (\tau)</td>
                <td>Conditional Average Treatment Effect: the causal impact of an action on a specific person's default probability.</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; font-weight: 700; color: #0F172A;">Agent-Based Model</td>
                <td>A computer simulation of individual people (agents) interacting in an economy over time to observe emergent macro behavior.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
