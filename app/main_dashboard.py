import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import joblib

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    SAVED_MODELS_DIR,
    REAL_MACRO_CSV,
    PROCESSED_MASTER_PARQUET,
    CUSTOMER_ONLY_FEATURES,
    MACRO_FEATURES
)
from models.feature_engineering import rank_features_by_information_gain
from models.explainability import ModelExplainabilityEngine

# ---------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Retail Credit Risk & Macro Regime Suite",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E232F;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #2B3345;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8C9BAB;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #E2E8F0;
        font-size: 26px;
        font-weight: 700;
    }
    .badge-pass {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_artifacts():
    metrics = joblib.load(SAVED_MODELS_DIR / "metrics.joblib")
    test_payload = joblib.load(SAVED_MODELS_DIR / "test_payload.joblib")
    macro_df = pd.read_csv(REAL_MACRO_CSV, index_col="date")
    master_df = pd.read_parquet(PROCESSED_MASTER_PARQUET)
    return metrics, test_payload, macro_df, master_df

metrics, test_payload, macro_df, master_df = load_all_artifacts()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/bank.png", width=64)
st.sidebar.title("Credit Risk Strategy")
st.sidebar.caption("Empirical Retail Credit & Macro Regime Prototype")

selected_tab = st.sidebar.radio(
    "Navigation",
    [
        "1. Head-to-Head Model Evaluation",
        "2. Macro Regimes & Stress Testing",
        "3. Information Gain & Entropy",
        "4. OSFI E-23 SHAP Explainability"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Integrity Check")
st.sidebar.info(
    f"**Real Customers:** {len(master_df):,}\n\n"
    f"**Test Sample:** {len(test_payload['y_test']):,}\n\n"
    f"**Empirical Default Rate:** {master_df['default_next_90_days'].mean():.2%}\n\n"
    f"**Macro History:** {len(macro_df)} Months"
)

# ---------------------------------------------------------
# TAB 1: Head-to-Head Model Evaluation
# ---------------------------------------------------------
if selected_tab == "1. Head-to-Head Model Evaluation":
    st.title("⚖️ Head-to-Head Model Comparison: Customer-Only vs. Macro-Aware")
    st.write(
        "A rigorous mathematical comparison between a standard borrower-only credit scoring model "
        "and a macroeconomic regime-aware XGBoost model, evaluated on out-of-fold calibrated probabilities."
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    m_base = metrics["customer_only"]
    m_macro = metrics["macro_aware"]
    
    with col1:
        st.metric(
            label="ROC-AUC (Macro-Aware)",
            value=f"{m_macro['roc_auc']:.4f}",
            delta=f"{m_macro['roc_auc'] - m_base['roc_auc']:+.4f} vs Baseline"
        )
    with col2:
        st.metric(
            label="PR-AUC (Macro-Aware)",
            value=f"{m_macro['pr_auc']:.4f}",
            delta=f"{m_macro['pr_auc'] - m_base['pr_auc']:+.4f} vs Baseline"
        )
    with col3:
        st.metric(
            label="Brier Score",
            value=f"{m_macro['brier_score']:.4f}",
            delta=f"{m_macro['brier_score'] - m_base['brier_score']:+.4f} (lower is better)",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="Expected Calib. Error (ECE)",
            value=f"{m_macro['ece']:.4f}",
            delta=f"{m_macro['ece'] - m_base['ece']:+.4f} (Decile dev)",
            delta_color="inverse"
        )
    with col5:
        st.metric(
            label="Log Loss",
            value=f"{m_macro['log_loss']:.4f}",
            delta=f"{m_macro['log_loss'] - m_base['log_loss']:+.4f} vs Baseline",
            delta_color="inverse"
        )

    st.markdown("---")
    
    # Detailed Curves in Tabs
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Receiver Operating Characteristic (ROC)", "Precision-Recall Curve", "Calibration Reliability Diagram"])
    
    y_test = test_payload["y_test"]
    prob_base = test_payload["prob_base"]
    prob_macro = test_payload["prob_macro"]
    
    with chart_tab1:
        fpr_b, tpr_b, _ = roc_curve(y_test, prob_base)
        fpr_m, tpr_m, _ = roc_curve(y_test, prob_macro)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_b, y=tpr_b, mode="lines", name=f"Customer-Only (AUC = {m_base['roc_auc']:.3f})", line=dict(color="#64748B", width=2)))
        fig_roc.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode="lines", name=f"Macro-Aware (AUC = {m_macro['roc_auc']:.3f})", line=dict(color="#38BDF8", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="#475569")))
        fig_roc.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            template="plotly_dark",
            height=480
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with chart_tab2:
        pr_b, rc_b, _ = precision_recall_curve(y_test, prob_base)
        pr_m, rc_m, _ = precision_recall_curve(y_test, prob_macro)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rc_b, y=pr_b, mode="lines", name=f"Customer-Only (PR-AUC = {m_base['pr_auc']:.3f})", line=dict(color="#64748B", width=2)))
        fig_pr.add_trace(go.Scatter(x=rc_m, y=pr_m, mode="lines", name=f"Macro-Aware (PR-AUC = {m_macro['pr_auc']:.3f})", line=dict(color="#34D399", width=3)))
        fig_pr.update_layout(
            title="Precision-Recall Curve (Imbalanced Default Outcome: 22%)",
            xaxis_title="Recall (Default Coverage)",
            yaxis_title="Precision (Positive Predictive Value)",
            template="plotly_dark",
            height=480
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with chart_tab3:
        frac_b, mean_b = calibration_curve(y_test, prob_base, n_bins=10, strategy="uniform")
        frac_m, mean_m = calibration_curve(y_test, prob_macro, n_bins=10, strategy="uniform")
        
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="#94A3B8")))
        fig_cal.add_trace(go.Scatter(x=mean_b, y=frac_b, mode="lines+markers", name="Customer-Only (Platt Calibrated)", line=dict(color="#F59E0B", width=2)))
        fig_cal.add_trace(go.Scatter(x=mean_m, y=frac_m, mode="lines+markers", name="Macro-Aware (Platt Calibrated)", line=dict(color="#818CF8", width=3)))
        fig_cal.update_layout(
            title="Reliability Diagram: Actual Default Frequency vs. Predicted Probability",
            xaxis_title="Mean Predicted Probability (Deciles)",
            yaxis_title="Observed Fraction of Defaults",
            template="plotly_dark",
            height=480
        )
        st.plotly_chart(fig_cal, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Macro Regimes & Stress Testing
# ---------------------------------------------------------
elif selected_tab == "2. Macro Regimes & Stress Testing":
    st.title("🌐 Macro Regimes & Dynamic Portfolio Stress Testing")
    st.write(
        "Real macroeconomic indicators from the **Bank of Canada Valet API** and **FRED**, "
        "processed through a **Hamilton (1989) Markov-Switching Autoregressive Engine**."
    )
    
    col1, col2, col3 = st.columns(3)
    latest_macro = macro_df.iloc[-1]
    with col1:
        st.metric("Bank of Canada Policy Rate", f"{latest_macro.get('policy_rate', 2.25):.2f}%", "Official Valet API")
    with col2:
        st.metric("US High Yield Credit Spread", f"{latest_macro.get('high_yield_spread', 2.93):.2f}%", "FRED BAMLH0A0HYM2")
    with col3:
        st.metric("Unemployment Rate Proxy", f"{latest_macro.get('unemployment_rate', 4.1):.1f}%", "National Bureau Series")
        
    st.markdown("---")
    
    # Macro Historical Charts
    fig_macro = go.Figure()
    fig_macro.add_trace(go.Scatter(x=macro_df.index, y=macro_df["high_yield_spread"], name="High Yield Spread (%)", line=dict(color="#F43F5E", width=2.5)))
    fig_macro.add_trace(go.Scatter(x=macro_df.index, y=macro_df["policy_rate"], name="BoC Policy Rate (%)", line=dict(color="#38BDF8", width=2)))
    fig_macro.update_layout(
        title="60-Month Real Macroeconomic Time Series",
        xaxis_title="Month",
        yaxis_title="Percent (%)",
        template="plotly_dark",
        height=380
    )
    st.plotly_chart(fig_macro, use_container_width=True)
    
    st.subheader("⚡ Live Macroeconomic Stress Testing Scenario")
    st.write("Simulate a macroeconomic shock and observe the instantaneous impact on predicted portfolio default distribution.")
    
    shock_spread = st.slider("Simulate High Yield Spread Shock (Basis Points)", min_value=0, max_value=600, value=150, step=25)
    shock_rate = st.slider("Simulate Central Bank Rate Hike (Basis Points)", min_value=0, max_value=300, value=75, step=25)
    
    # Calculate shifted risk probabilities
    macro_model = joblib.load(SAVED_MODELS_DIR / "macro_model.joblib")
    macro_pipeline = joblib.load(SAVED_MODELS_DIR / "macro_pipeline.joblib")
    
    X_sample = test_payload["X_macro_test"].head(1000).copy()
    baseline_pd = test_payload["prob_macro"][:1000]
    
    # Apply shock to feature columns
    X_shocked = X_sample.copy()
    X_shocked["high_yield_spread"] += shock_spread / 100.0
    X_shocked["policy_rate"] += shock_rate / 100.0
    X_shocked["systemic_risk_index"] = np.clip(X_shocked["systemic_risk_index"] + (shock_spread / 400.0), 0.0, 1.0)
    
    X_trans_shocked = macro_pipeline.transform(X_shocked)
    shocked_pd = macro_model.predict_proba(X_trans_shocked)[:, 1]
    
    st.markdown(f"**Mean Portfolio Default Probability:** {baseline_pd.mean():.2%} ➡️ <span style='color:#F43F5E; font-size:18px; font-weight:bold;'>{shocked_pd.mean():.2%}</span> (Shift: +{(shocked_pd.mean() - baseline_pd.mean())*100:.2f}%)", unsafe_allow_html=True)
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=baseline_pd, nbinsx=30, name="Baseline Regime", marker_color="#38BDF8", opacity=0.7))
    fig_dist.add_trace(go.Histogram(x=shocked_pd, nbinsx=30, name="Stressed Macro Regime", marker_color="#F43F5E", opacity=0.7))
    fig_dist.update_layout(
        barmode="overlay",
        title="Portfolio Default Probability Shift under Stressed Conditions",
        xaxis_title="Predicted Probability of Default (PD)",
        yaxis_title="Borrower Count",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Information Gain & Entropy
# ---------------------------------------------------------
elif selected_tab == "3. Information Gain & Entropy":
    st.title("🧮 Shannon Information Entropy & Feature Information Gain")
    st.write(
        "Mathematically screen predictive power without model bias using "
        "Shannon Information Entropy $H(Y)$ and Information Gain $IG(Y, X) = H(Y) - H(Y|X)$."
    )
    
    all_features = CUSTOMER_ONLY_FEATURES + MACRO_FEATURES
    ranks = rank_features_by_information_gain(master_df, all_features)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_ig = px.bar(
            ranks,
            x="information_gain",
            y="feature",
            orientation="h",
            color="information_gain",
            color_continuous_scale="Viridis",
            title="Feature Ranking by Shannon Information Gain (Bits)"
        )
        fig_ig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", height=500)
        st.plotly_chart(fig_ig, use_container_width=True)
    with col2:
        st.subheader("Information Gain Ranking Table")
        st.dataframe(ranks.style.format({"information_gain": "{:.5f}"}), height=480, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: OSFI E-23 SHAP Explainability
# ---------------------------------------------------------
elif selected_tab == "4. OSFI E-23 SHAP Explainability":
    st.title("🛡️ OSFI Guideline E-23 Model Governance & SHAP Explainability")
    st.write(
        "Canadian bank regulatory compliance requires decomposing black-box credit decisions into "
        "axiomatic Shapley attributions (Lundberg & Lee 2017) to provide auditable adverse-action reasons."
    )
    
    explainer_engine = ModelExplainabilityEngine("macro_aware")
    
    subtab1, subtab2 = st.tabs(["Individual Customer Adverse Action Notice", "Global Portfolio SHAP Feature Importance"])
    
    with subtab1:
        st.subheader("Select a Customer to Audit")
        customer_ids = test_payload["customer_ids"][:50].tolist()
        selected_cust_id = st.selectbox("Customer Account ID", customer_ids)
        
        cust_idx = customer_ids.index(selected_cust_id)
        explanation = explainer_engine.generate_local_explanation(cust_idx)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Customer ID", f"#{explanation['customer_id']}")
        with c2:
            st.metric("Model Baseline (Average)", f"{explanation['base_value']:.2f} (Log-odds)")
        with c3:
            pd_val = explanation["predicted_pd"]
            color = "inverse" if pd_val > 0.25 else "normal"
            st.metric("Calibrated 90-Day PD", f"{pd_val:.2%}")
            
        st.markdown("#### Top Risk Factors (Adverse Action Decomposition)")
        top_contribs = explanation["contributions"].head(10).copy()
        top_contribs["Direction"] = top_contribs["shap_value"].apply(lambda v: "🔴 Increases Risk" if v > 0 else "🟢 Reduces Risk")
        
        fig_waterfall = px.bar(
            top_contribs,
            x="shap_value",
            y="feature",
            orientation="h",
            color="Direction",
            color_discrete_map={"🔴 Increases Risk": "#F43F5E", "🟢 Reduces Risk": "#10B981"},
            title=f"SHAP Log-Odds Contributions for Customer #{explanation['customer_id']}"
        )
        fig_waterfall.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", height=420)
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown("#### Full Profile & Regulatory Audit Trail")
        st.json(explanation["raw_features"])
        
    with subtab2:
        st.subheader("Global Portfolio-Wide Feature Attributions")
        global_shap = explainer_engine.get_global_shap_importance(500)
        
        fig_glob = px.bar(
            global_shap.head(15),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Teal",
            title="Global Mean Absolute SHAP Importance (|SHAP|)"
        )
        fig_glob.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", height=500)
        st.plotly_chart(fig_glob, use_container_width=True)
