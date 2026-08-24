
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="RiskGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("riskguard_model.pkl")
scaler = joblib.load("riskguard_scaler.pkl")
config = joblib.load("riskguard_config.pkl")

fraud_threshold = config["fraud_threshold"]

# ============================================
# FUNCTIONS
# ============================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 70:
        return "MEDIUM"

    else:
        return "HIGH"


def get_recommended_action(score):

    if score < 30:
        return "APPROVE"

    elif score < 70:
        return "MANUAL REVIEW"

    else:
        return "BLOCK / VERIFY"


def analyze_transactions(dataframe):

    required_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    model_data = dataframe[
        required_columns
    ].copy()

    scaled_data = scaler.transform(
        model_data
    )

    fraud_probabilities = model.predict_proba(
        scaled_data
    )[:, 1]

    risk_scores = (
        fraud_probabilities * 100
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
        fraud_probabilities.round(4)
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

st.sidebar.header("🛡️ RiskGuard AI")

st.sidebar.write(
    "Machine-learning powered payment "
    "risk assessment."
)

st.sidebar.divider()

st.sidebar.metric(
    "Fraud Threshold",
    f"{fraud_threshold:.2f}"
)

st.sidebar.write(
    "Risk Score Bands"
)

st.sidebar.write(
    "🟢 0–30 → Low Risk"
)

st.sidebar.write(
    "🟡 30–70 → Medium Risk"
)

st.sidebar.write(
    "🔴 70–100 → High Risk"
)

# ============================================
# FILE UPLOAD
# ============================================

st.header("📁 Upload Transactions")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

# ============================================
# PROCESS FILE
# ============================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"Uploaded successfully — "
            f"{len(df):,} transactions found."
        )

        # -----------------------------
        # Preview
        # -----------------------------

        st.subheader(
            "Transaction Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # -----------------------------
        # Analyze
        # -----------------------------

        if st.button(
            "🔍 ANALYZE TRANSACTIONS",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing transactions..."
            ):

                results = analyze_transactions(
                    df
                )

            st.success(
                "✅ Analysis completed!"
            )

            st.divider()

            # =================================
            # RISK SUMMARY
            # =================================

            st.header(
                "📊 Risk Summary"
            )

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
                    f"{len(results):,}"
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
            # HIGH RISK ALERT
            # =================================

            if high_count > 0:

                st.error(
                    f"🚨 {high_count:,} "
                    "high-risk transaction(s) "
                    "detected."
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

            available_columns = [
                col
                for col in display_columns
                if col in results.columns
            ]

            st.dataframe(
                results[available_columns],
                use_container_width=True,
                hide_index=True
            )

            # =================================
            # HIGH-RISK TRANSACTIONS
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
                    high_risk[available_columns],
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No high-risk transactions found."
                )

            # =================================
            # DOWNLOAD RESULTS
            # =================================

            st.divider()

            csv_output = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Risk Report",
                data=csv_output,
                file_name="riskguard_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================
# NO FILE
# ============================================

else:

    st.info(
        "👆 Upload a transaction CSV file "
        "to begin risk analysis."
    )

    st.write(
        "The uploaded CSV must contain the "
        "30 features used by the trained model."
    )

# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "RiskGuard AI | AI Risk Management Prototype | "
    "Razorpay AI Buildathon 2026"
)
