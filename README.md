# 🤖 AI-Powered Customer Churn Prediction System

Machine Learning-powered Customer Analytics Application for predicting customer churn using demographic and behavioral data. Built with Python, Scikit-learn, XGBoost, and Streamlit to help businesses identify at-risk customers and improve customer retention strategies.

**Python • Scikit-learn • XGBoost • Streamlit • Pandas • NumPy**

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by businesses. Retaining existing customers is often more cost-effective than acquiring new ones.

This project leverages Machine Learning techniques to predict whether a customer is likely to leave a service based on historical customer data and behavioral patterns.

The application provides:

* Real-time churn prediction
* Probability-based churn assessment
* Interactive Streamlit dashboard
* Customer behavior analysis
* Data preprocessing and feature engineering pipeline
* Business decision support for retention strategies

---

## 🏗️ System Architecture

Customer Demographic & Behavioral Data 
        │
        ▼
Data Preprocessing Pipeline (Cleaning, Encoding, Validation) 
        │
        ▼
Feature Engineering & Scaling 
        │
        ▼
Machine Learning Prediction Engine (XGBoost Classifier)
        │
        ▼
Churn Probability Estimation 
        │
        ▼
Customer Risk Classification
        │
        ▼
Interactive Streamlit Dashboard 
        │
        ▼
Actionable Business Retention Insights

---

## 🛠️ Tech Stack

| Category                 | Technology                  |
| ------------------------ | --------------------------- |
| Programming Language     | Python                      |
| Machine Learning         | Scikit-learn, XGBoost       |
| Frontend                 | Streamlit                   |
| Data Processing          | Pandas, NumPy               |
| Visualization            | Matplotlib, Seaborn, Plotly |
| Model Persistence        | Joblib                      |
| Imbalanced Data Handling | SMOTE                       |

---

## 📁 Repository Structure

```text
AI_POWERED_CUSTOMER_CHURN_PREDICTION_SYSTEM/
│
├── churn_app.py
├── train_model.py
├── customer_churn_prediction.ipynb
│
├── dataset/
│   └── customer_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── assets/
    └── screenshots/
```

## 📊 Dataset Information

The project uses customer demographic and behavioral data to predict churn.

### Features

* Gender
* Senior Citizen
* Partner Status
* Dependents
* Tenure
* Internet Service
* Contract Type
* Monthly Charges
* Total Charges
* Payment Method
* Online Security
* Tech Support
* Streaming Services
* Multiple Lines

### Target Variable

| Value | Meaning           |
| ----- | ----------------- |
| 0     | Customer Retained |
| 1     | Customer Churned  |

---

## ⚙️ Machine Learning Pipeline

### 1. Data Cleaning

* Handle missing values
* Remove inconsistencies
* Prepare dataset for training

### 2. Feature Engineering

* Label Encoding
* One-Hot Encoding
* Feature Selection

### 3. Feature Scaling

* StandardScaler()

### 4. Class Balancing

* SMOTE()

### 5. Model Training

Algorithms explored:

* Logistic Regression
* Random Forest
* XGBoost ✅
* Decision Tree

### 6. Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

---

## 🤖 Trained Model

### Production Model

* XGBoost Classifier

### Saved Artifacts

* churn_model.pkl
* scaler.pkl
* columns.pkl

---

## 📈 Application Features

### Customer Churn Prediction

Predicts whether a customer is likely to churn.

### Churn Probability Score

Provides confidence level of prediction.

### Interactive Dashboard

Built using Streamlit:

* User-friendly interface
* Real-time prediction
* Customer insights
* Business-friendly visualizations

### Data Visualization

* Customer distribution analysis
* Churn trends
* Feature importance analysis
* Interactive charts using Plotly

---

## 📷 Application Screenshots

### Dashboard Home

*Add screenshot here*

### Prediction Results

*Add screenshot here*

### Customer Insights

*Add screenshot here*

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/AI_POWERED_CUSTOMER_CHURN_PREDICTION_SYSTEM.git

cd AI_POWERED_CUSTOMER_CHURN_PREDICTION_SYSTEM
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run churn_app.py
```

---

## 📷 Application Workflow

Enter Customer Information

│

▼

Click Predict

│

▼

Model Processes Data

│

▼

Churn Probability Generated

│

▼

Prediction Result Displayed

│

▼

Business Insights Provided

---

## 🎯 Future Improvements

* Deep Learning Models
* Customer Lifetime Value Prediction
* Real-time API Integration
* Personalized Retention Recommendations
* Cloud Deployment
* Automated Retraining Pipeline
* Advanced Customer Segmentation
* Explainable AI Integration (SHAP)

---

## ⚠️ Disclaimer

This project is developed for educational, research, and business analytics purposes. Predictions generated by the model should be used as decision-support insights and not as the sole basis for business actions.
