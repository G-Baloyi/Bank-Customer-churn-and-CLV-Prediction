# Bank Customer Churn & CLV Prediction

A machine learning project that predicts **customer churn**, estimates **Customer Lifetime Value (CLV)**, and identifies **high-priority customers for retention** using a dataset of **50,000+ customers and 19+ features**.

---

## Project Overview

This project combines **classification, regression, clustering, and explainable AI** to solve a real-world business problem:

* Predict **which customers will churn**
* Estimate **how valuable each customer is (CLV)**
* Prioritize **which customers to retain first**
* Provide **business insights using visualizations and SHAP explainability**

---

## Dataset

* **Size**: 50,000+ rows, 19+ columns
* **Type**: Bank customer data
* **Key Features**:

  * Demographics (Gender, Region)
  * Financial data (Balance, Salary)
  * Behavior (Tenure, Products, Activity)
  * Target variables:

    * `Exited` → Churn (0/1)
    * `CLV` → Customer Lifetime Value

---

## Machine Learning Pipeline

### 1. Data Cleaning

* Missing values handled using:

  * Median (numerical)
  * Mode / fallback values (categorical)
* Automatic conversion of numeric-like strings

---

### 2. Feature Engineering

* Created advanced features:

  * `Balance_to_Salary`
  * `Balance_Tenure`
  * `Products_Active`
* Customer segmentation using **KMeans Clustering**
* Tenure grouped into categories
* One-hot encoding for categorical variables

---

### 3. Models

#### Churn Prediction (Classification)

Ensemble model using:

* Random Forest
* Gradient Boosting
* Logistic Regression
   Combined using **VotingClassifier**

**Metrics:**

* Accuracy
* ROC-AUC
* Classification Report

---

#### CLV Prediction (Regression)

* Random Forest Regressor

**Metrics:**

* RMSE
* R² Score

---

### 4. Retention Priority Score

A key business metric:

```
Retention Score = Churn Probability × Predicted CLV
```

 Identifies **high-value customers at risk of leaving**

---

## Visualizations

### Interactive (Plotly)

* Churn vs CLV scatter plot
* Top 20 high-priority customers

### Statistical (Seaborn / Matplotlib)

* Churn distribution
* CLV distribution
* Correlation heatmap

---

## Explainable AI (SHAP)

* Used **SHAP (SHapley Additive exPlanations)** to interpret model predictions
* Helps answer:

  * Why a customer is predicted to churn
  * Which features influence CLV

---

## Results

* Accurate churn prediction using ensemble learning
* Reliable CLV estimation
* Clear identification of:

  * High-risk customers
  * High-value customers
* Business-ready **Retention Priority Score**

---

## Future Improvements

* Deploy as a web app (Streamlit / Flask)
* Real-time prediction API
* Dashboard for business users
* Model optimization (XGBoost, LightGBM)

---
