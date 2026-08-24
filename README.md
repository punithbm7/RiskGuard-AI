
# 🛡️ RiskGuard AI

## AI-Powered Payment Fraud & Risk Detection

RiskGuard AI is a machine-learning powered payment risk detection system designed to identify potentially fraudulent transactions and provide an actionable risk assessment.

The system combines machine learning, risk scoring, threshold optimization, and explainable AI to help identify suspicious payment activity.

---

## 🚀 Key Features

- 🤖 Machine-learning based fraud detection
- 📊 Random Forest and Logistic Regression comparison
- 🎯 Fraud probability estimation
- 🛡️ 0–100 transaction risk score
- 🟢 Low / 🟡 Medium / 🔴 High risk classification
- 🚨 Recommended payment action
- 🧠 SHAP-based model explainability
- 📁 Batch CSV transaction analysis
- 📥 Downloadable risk reports
- 📊 Transaction risk analytics

---

## 🧠 System Architecture

```text
Payment Transactions
        │
        ▼
Data Validation
        │
        ▼
Feature Scaling
        │
        ▼
Machine Learning Model
        │
        ▼
Fraud Probability
        │
        ▼
Risk Score (0–100)
        │
        ▼
Risk Classification
        │
   ┌────┼────┐
   ▼    ▼    ▼
 LOW  MEDIUM HIGH
   │    │    │
   ▼    ▼    ▼
APPROVE REVIEW BLOCK/VERIFY
        │
        ▼
SHAP Explainability
