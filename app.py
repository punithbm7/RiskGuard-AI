
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="RiskGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "riskguard_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "riskguard_scaler.pkl"
CONFIG_PATH = BASE_DIR / "models" / "riskguard_config.pkl"


# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
config = joblib.load(CONFIG_PATH)

fraud_threshold = config["fraud_threshold"]


# ============================================
# RISK FUNCTIONS
# ============================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    if score < 70:
        return "MEDIUM"

    return "HIGH"


def get_recommended_action(score):

    if score < 30:
        return "APPROVE"

    if score < 70:
        return "MANUAL REVIEW"

    return "BLOCK / VERIFY"


# ============================================
# ANALYZE TRANSACTIONS
# ============================================

def analyze_transactions(dataframe):

    required_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    model_data = dataframe[
        required_columns
    ].copy()

    scaled_data = scaler.transform(
        model_data
    )

    probabilities = model.predict_proba(
        scaled_data
    )[:, 1]

    risk_scores = (
        probabilities * 100
    ).round(2)

    risk_levels = [
        get_risk_level(score)
        for score in risk_scores
    ]

    actions = [
        get_recommended_action(score)
        for score in risk_scores
    ]

    results = dataframe.copy()

    results["Fraud Probability"] = (
        probabilities.round(4)
    )

    results["Risk Score"] = risk_scores

    results["Risk Level"] = risk_levels

    results["Recommended Action"] = actions

    return results


# ============================================
# HEADER
# ============================================

st.title("🛡️ RiskGuard AI")

st.subheader(
    "AI-Powered Payment Fraud & Risk Detection"
)

st.write(
    "Upload payment transactions and let "
    "RiskGuard identify potentially risky activity."
)

st.divider()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🛡️ RiskGuard AI")

st.sidebar.write(
    "Machine-learning powered payment "
    "risk assessment."
)

st.sidebar.divider()

st.sidebar.metric(
    "Fraud Threshold",
    f"{fraud_threshold:.2f}"
)

st.sidebar.write("Risk Score Bands")

st.sidebar.write("🟢 0–30 → Low Risk")
st.sidebar.write("🟡 30–70 → Medium Risk")
st.sidebar.write("🔴 70–100 → High Risk")


# ============================================
# FILE UPLOAD
# ============================================

st.header("📁 Upload Transactions")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


# ============================================
# NO FILE
# ============================================

if uploaded_file is None:

    st.info(
        "👆 Upload a transaction CSV file "
        "to begin risk analysis."
    )

    st.write("Required CSV columns:")

    st.code(
        "Time, V1, V2, ..., V28, Amount"
    )


# ============================================
# FILE UPLOADED
# ============================================

if uploaded_file is not None:

    try:

        dataframe = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"Uploaded successfully — "
            f"{len(dataframe):,} transactions found."
        )

        st.subheader(
            "Transaction Preview"
        )

        st.dataframe(
            dataframe.head(10),
            use_container_width=True
        )

        analyze_button = st.button(
            "🔍 ANALYZE TRANSACTIONS",
            use_container_width=True
        )

        if analyze_button:

            with st.spinner(
                "Analyzing transactions..."
            ):

                results = analyze_transactions(
                    dataframe
                )

            st.success(
                "✅ Analysis completed!"
            )

            st.divider()

            # =================================
            # SUMMARY
            # =================================

            st.header("📊 Risk Summary")

            total_count = len(results)

            low_count = (
                results["Risk Level"]
                .eq("LOW")
                .sum()
            )

            medium_count = (
                results["Risk Level"]
                .eq("MEDIUM")
                .sum()
            )

            high_count = (
                results["Risk Level"]
                .eq("HIGH")
                .sum()
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Total Transactions",
                    f"{total_count:,}"
                )

            with col2:

                st.metric(
                    "🟢 Low Risk",
                    f"{low_count:,}"
                )

            with col3:

                st.metric(
                    "🟡 Medium Risk",
                    f"{medium_count:,}"
                )

            with col4:

                st.metric(
                    "🔴 High Risk",
                    f"{high_count:,}"
                )

            # =================================
            # ALERT
            # =================================

            if high_count > 0:

                st.error(
                    f"🚨 {high_count:,} "
                    "high-risk transaction(s) detected."
                )

            else:

                st.success(
                    "✅ No high-risk transactions detected."
                )

            # =================================
            # RESULTS
            # =================================

            st.divider()

            st.header(
                "🔎 Transaction Risk Results"
            )

            display_columns = [
                "Amount",
                "Fraud Probability",
                "Risk Score",
                "Risk Level",
                "Recommended Action"
            ]

            st.dataframe(
                results[display_columns],
                use_container_width=True,
                hide_index=True
            )

            # =================================
            # HIGH RISK
            # =================================

            st.divider()

            st.header(
                "🚨 High-Risk Transactions"
            )

            high_risk = results[
                results["Risk Level"] == "HIGH"
            ]

            if len(high_risk) > 0:

                st.dataframe(
                    high_risk[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No high-risk transactions found."
                )

            # =================================
            # DOWNLOAD
            # =================================

            st.divider()

            report = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Risk Report",
                data=report,
                file_name="riskguard_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as error:

        st.error(
            "❌ Unable to analyze the uploaded file."
        )

        st.exception(error)


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "RiskGuard AI | AI Risk Management Prototype | "
    "Razorpay AI Buildathon 2026"
)
