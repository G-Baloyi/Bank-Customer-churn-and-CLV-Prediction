# Bank Customer Churn & CLV Prediction

> **A machine learning pipeline that predicts customer churn, estimates Customer Lifetime Value (CLV), and identifies high-priority customers for targeted retention campaigns.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-brightgreen)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-blue)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-red)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Customer churn is one of the most costly problems in retail banking. This project builds a **full end-to-end ML solution** that:

- Performs rich **exploratory data analysis** on 50,000 bank customers
- Trains and compares **three churn classifiers** (Logistic Regression, Random Forest, XGBoost)
- Predicts **Customer Lifetime Value** with regression models
- Segments all customers into a **Risk × Value retention matrix**
- Explains model decisions with **SHAP** feature importance

---

## Dataset

| Property | Value |
|---|---|
| Rows | 50,000 customers |
| Features | 16 columns |
| Churn Rate | ~24.6% |
| CLV Range | $0.20 – $25,852 |
| Missing Values | None |

### Feature Overview

| Feature | Type | Description |
|---|---|---|
| `CustomerID` | ID | Unique customer identifier |
| `Age` | Numeric | Customer age |
| `Gender` | Categorical | Male / Female |
| `Tenure` | Numeric | Years as a customer |
| `Balance` | Numeric | Account balance ($) |
| `NumOfProducts` | Numeric | Number of bank products held |
| `HasCreditCard` | Binary | Credit card holder? |
| `IsActiveMember` | Binary | Active account? |
| `EstimatedSalary` | Numeric | Estimated annual salary ($) |
| `CreditScore` | Numeric | Credit score (300–850) |
| `TotalTransactions` | Numeric | Total transaction count |
| `AverageTransactionValue` | Numeric | Mean transaction value ($) |
| `AccountType` | Categorical | Standard / Premium |
| `Region` | Categorical | North / South / East / West |
| `Exited` | **Target (Churn)** | 0 = Retained, 1 = Churned |
| `CLV` | **Target (CLV)** | Customer Lifetime Value ($) |

---

## Project Structure

```
bank-churn-clv-prediction/
│
├── Bank_Customer_Churn_CLV_Prediction.ipynb   # Main notebook
├── bank_customers_large.csv                   # Dataset (50k rows)
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
└── reports/
    └── Project_Report.docx                    # Full project report
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bank-churn-clv-prediction.git
cd bank-churn-clv-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook Bank_Customer_Churn_CLV_Prediction.ipynb
```

---

## Usage

Simply open the notebook and **Run All Cells** (`Kernel → Restart & Run All`).

The notebook is fully self-contained. Update the dataset path in **Section 2** if needed:

```python
df = pd.read_csv('bank_customers_large.csv')
```

---

## Methodology

### 1. Feature Engineering
Five engineered features are derived from the raw data:

| Feature | Formula / Logic |
|---|---|
| `BalancePerProduct` | Balance ÷ (NumOfProducts + 1) |
| `EngagementScore` | Weighted combination of transactions, avg value, active status |
| `AgeGroup` | Binned: Young / Middle / Senior / Elderly |
| `TenureGroup` | Binned: New / Growing / Mature / Loyal |
| `CreditBand` | Binned: Poor / Fair / Good / Excellent |

### 2. Churn Classification
Three classifiers trained and compared:
- **Logistic Regression** — regularised linear baseline with `class_weight='balanced'`
- **Random Forest** — 200-tree ensemble with balanced class weights
- **XGBoost** — gradient boosting with `scale_pos_weight` for imbalance handling

### 3. CLV Regression
Two regressors compared:
- **Ridge Regression** — L2-regularised linear model (α = 10)
- **Gradient Boosting Regressor** — 200-estimator boosted ensemble

### 4. SHAP Explainability
TreeExplainer applied to XGBoost on a 2,000-sample test subset to produce:
- Beeswarm summary plot (global feature impact)
- Bar plot (mean absolute SHAP values)

### 5. Retention Segmentation
All 50,000 customers segmented using a **2×2 Risk × CLV matrix**:

```
High CLV │  Priority 3 — Valuable  │  Priority 1 — Critical  │
         │  (Low Risk, High CLV)   │  (High Risk, High CLV)  │
─────────┼─────────────────────────┼─────────────────────────┤
Low CLV  │  Priority 4 — Standard  │  Priority 2 — At Risk   │
         │  (Low Risk, Low CLV)    │  (High Risk, Low CLV)   │
         └─────────────────────────┴─────────────────────────┘
                    Low Churn Risk           High Churn Risk
```

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | ML models, preprocessing, evaluation |
| `xgboost` | ≥ 1.7 | Gradient boosting classifier |
| `shap` | ≥ 0.44 | Model explainability |
| `matplotlib` | ≥ 3.7 | Plotting |
| `seaborn` | ≥ 0.12 | Statistical visualisation |
| `jupyter` | ≥ 1.0 | Notebook environment |

---
