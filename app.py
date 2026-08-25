
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sklearn
import numpy
# ============================================
# DEPLOYMENT VERSION
# ============================================

DEPLOYMENT_VERSION = "RiskGuard-CLEAN-RF-2026-08-25"

# ============================================
# PAGE CONFIG
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

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "riskguard_model.pkl"
)

AMOUNT_SCALER_PATH = (
    BASE_DIR /
    "models" /
    "riskguard_amount_scaler.pkl"
)

TIME_SCALER_PATH = (
    BASE_DIR /
    "models" /
    "riskguard_time_scaler.pkl"
)

CONFIG_PATH = (
    BASE_DIR /
    "models" /
    "riskguard_config.pkl"
)


# ============================================
# LOAD ML PIPELINE
# ============================================

model = joblib.load(
    MODEL_PATH
)

amount_scaler = joblib.load(
    AMOUNT_SCALER_PATH
)

time_scaler = joblib.load(
    TIME_SCALER_PATH
)

config = joblib.load(
    CONFIG_PATH
)

feature_columns = config[
    "feature_columns"
]

fraud_threshold = config[
    "fraud_threshold"
]

# ============================================
# FILE FINGERPRINT DIAGNOSTICS
# ============================================

import hashlib

def file_hash(path):

    with open(path, "rb") as f:
        return hashlib.sha256(
            f.read()
        ).hexdigest()


MODEL_HASH = file_hash(
    MODEL_PATH
)

AMOUNT_SCALER_HASH = file_hash(
    AMOUNT_SCALER_PATH
)

TIME_SCALER_HASH = file_hash(
    TIME_SCALER_PATH
)

CONFIG_HASH = file_hash(
    CONFIG_PATH
)


# ============================================
# RISK FUNCTIONS
# ============================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 70:
        return "MEDIUM"

    else:
        return "HIGH"


def get_action(level):

    if level == "LOW":
        return "APPROVE"

    elif level == "MEDIUM":
        return "MANUAL REVIEW"

    return "BLOCK / VERIFY"


# ============================================
# PREDICTION FUNCTION
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

    model_input = dataframe[
        required_columns
    ].copy()

    st.write("DEBUG - Raw first transaction:")
    st.write(dataframe.iloc[0].to_dict())

    st.write("DEBUG - Raw first 10 Amount values:")
    st.write(dataframe["Amount"].head(10).tolist())

    st.write("DEBUG - Raw first 10 Time values:")
    st.write(dataframe["Time"].head(10).tolist())

    # ----------------------------------------
    # Exact training preprocessing
    # ----------------------------------------

    model_input[
        "scaled_amount"
    ] = amount_scaler.transform(
        model_input[
            "Amount"
        ].values.reshape(-1, 1)
    ).flatten()

    model_input[
        "scaled_time"
    ] = time_scaler.transform(
        model_input[
            "Time"
        ].values.reshape(-1, 1)
    ).flatten()

    # ----------------------------------------
    # Exact feature order
    # ----------------------------------------

    X_input = model_input[
        feature_columns
    ]

    # ----------------------------------------
    # Fraud probability
    # ----------------------------------------

    probabilities = model.predict_proba(
        X_input
    )[:, 1]

    # ========================================
# DEBUG PREDICTIONS
# ========================================
    
    st.write("DEBUG - First 10 fraud probabilities:")
    st.write(probabilities[:10])
    
    st.write("DEBUG - First 10 risk scores:")
    st.write((probabilities[:10] * 100).round(2))
    
    st.write("DEBUG - Input shape:")
    st.write(X_input.shape)
    
    st.write("DEBUG - First transaction model input:")
    st.write(X_input.iloc[0].to_dict())
    # ----------------------------------------
    # Risk score
    # ----------------------------------------

    risk_scores = (
        probabilities * 100
    ).round(2)

    risk_levels = [
        get_risk_level(score)
        for score in risk_scores
    ]

    actions = [
        get_action(level)
        for level in risk_levels
    ]

    # ----------------------------------------
    # Results
    # ----------------------------------------

    results = dataframe.copy()

    results[
        "Fraud Probability"
    ] = probabilities.round(4)

    results[
        "Risk Score"
    ] = risk_scores

    results[
        "Risk Level"
    ] = risk_levels

    results[
        "Recommended Action"
    ] = actions

    return results


# ============================================
# HEADER
# ============================================

st.title(
    "🛡️ RiskGuard AI"
)
st.caption(
    "Deployment: RiskGuard-CLEAN-RF-2026-08-25"
)

with st.expander("🔧 Model Diagnostics"):
    st.write("Model:", type(model).__name__)
    st.write("Number of trees:", model.n_estimators)
    st.write("Model features:", model.n_features_in_)
    st.write("Amount scaler:", type(amount_scaler).__name__)
    st.write("Time scaler:", type(time_scaler).__name__)

    st.write("scikit-learn:", sklearn.__version__)
    st.write("NumPy:", numpy.__version__)
    st.write("Joblib:", joblib.__version__)
    st.write("Pandas:", pd.__version__)

with st.expander("🔐 File Fingerprints"):

    st.write(
        "Model SHA256:",
        MODEL_HASH
    )

    st.write(
        "Amount scaler SHA256:",
        AMOUNT_SCALER_HASH
    )

    st.write(
        "Time scaler SHA256:",
        TIME_SCALER_HASH
    )

    st.write(
        "Config SHA256:",
        CONFIG_HASH
    )

st.subheader(
    "AI-Powered Payment Fraud & Risk Detection"
)

st.write(
    "Upload payment transactions and "
    "RiskGuard will assess fraud probability "
    "and transaction risk."
)

st.divider()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title(
    "🛡️ RiskGuard AI"
)

st.sidebar.write(
    "Machine-learning powered "
    "payment risk assessment."
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
# UPLOAD
# ============================================

st.header(
    "📁 Upload Transactions"
)

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

    st.write(
        "Required columns:"
    )

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

                results = (
                    analyze_transactions(
                        dataframe
                    )
                )

            st.success(
                "✅ Analysis completed!"
            )

            # =================================
            # SUMMARY
            # =================================

            st.divider()

            st.header(
                "📊 Risk Summary"
            )

            total_count = len(results)

            low_count = (
                results[
                    "Risk Level"
                ] == "LOW"
            ).sum()

            medium_count = (
                results[
                    "Risk Level"
                ] == "MEDIUM"
            ).sum()

            high_count = (
                results[
                    "Risk Level"
                ] == "HIGH"
            ).sum()

            col1, col2, col3, col4 = (
                st.columns(4)
            )

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
                results[
                    display_columns
                ],
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
                results[
                    "Risk Level"
                ] == "HIGH"
            ]

            if len(high_risk) > 0:

                st.dataframe(
                    high_risk[
                        display_columns
                    ],
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
            ).encode(
                "utf-8"
            )

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
