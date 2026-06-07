# 🛡️ Real-Time Credit Card Fraud Detection System

An end-to-end, production-ready Machine Learning system engineered to detect and block fraudulent credit card transactions in real-time. Built using Python, Scikit-Learn, and clean object-oriented architecture, this project handles extreme class imbalance and financial outliers to deliver highly generalized, accurate predictions.

---

## 📌 Project Overview

### ⚠️ The Problem
The exponential growth of online transactions has triggered a surge in sophisticated financial fraud. Traditional security systems rely on rigid, rule-based logic that fails to adapt to evolving attack patterns. This leaves financial institutions vulnerable to massive monetary losses while increasing "false positives" that frustrate legitimate users and harm customer trust.

### 🎯 The Objective
To design and deploy an intelligent, machine-learning-driven system that accurately detects and blocks fraudulent credit card transactions in real time. The system focuses on achieving high precision and recall while minimizing friction for legitimate consumers.

### 💼 Operational Impact
* **Saves Money:** Directly eliminates financial exposure to stolen funds, legal penalties, and chargeback fees.
* **Builds Trust:** Secures user data seamlessly without interrupting the friction-free shopping experience.
* **Cuts Costs:** Automates complex security workflows, significantly reducing human reliance on manual review queues.

---

## 📊 Dataset Description

* **Source:** Benchmarked against the authoritative [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) (European Cardholders).
* **Timeline & Samples:** Captures 284,807 transactions spanning a two-day period.
* **Extreme Class Imbalance:** Features an incredibly sparse fraud distribution—**only 492 fraud cases out of 284,807 total transactions (~0.17%)**.
* **Anonymized Features:** * `V1` through `V28`: Numeric dimensions derived via Principal Component Analysis (PCA).
  * `Time` & `Amount`: Raw, untransformed transaction tracking attributes.
  * `Class` (Target Variable): Binary label where `1` indicates Fraud and `0` indicates Legitimate.

### Data Splitting Strategy
The dataset was structured into distinct files courtesy of authorized project stakeholders:
* **Training Set (`train.csv`):** 170,884 samples allocated exclusively to model training.
* **Validation (`val.csv`) & Test Sets (`test.csv`):** The remaining samples partitioned evenly to facilitate iterative evaluation and final independent testing.

> 📝 **Exploratory Data Analysis (EDA):** Deep cleaning operations, feature distributions, and custom feature engineering methodologies are documented inside this repository under `reports/read my insights.docx`.

---

## ⚙️ Data Preprocessing Pipeline

To combat data anomalies and prepare features for optimal algorithm performance, a customized, reusable pipeline class (`FraudPreprocessor`) was engineered:

1. **Robust Scaling (`RobustScaler`):** Applied exclusively to the `Amount` variable instead of standard normalization ($Z$-score) or MinMax scaling. Because transaction data naturally contains extreme financial outliers, `RobustScaler` utilizes the median and Interquartile Range (IQR) to safely normalize data without shrinking the mathematical presence of fraud.
2. **Median Imputation (`SimpleImputer`):** Automatically targets potential missing values across all dimensions using a `median` strategy, preventing downstream estimator crashes while maintaining statistical data distributions.

---

## 🔬 Modeling & Experimental Results

### 1. Baseline Model: Logistic Regression
* **Verdict:** Severely underfits the data. The model is too simple to capture the complex, non-linear boundaries separating sparse fraud samples from normal credit card activity.
* **Validation Results:**
  * **Precision:** `0.05` (Yielded a very high volume of false alarms)
  * **Recall:** `0.89`
  * **F1-Score:** `0.09195`

### 2. Advanced Model: RandomForestClassifier
* **Configuration:** Configured with `class_weight='balanced'` to explicitly force the decision trees to prioritize the highly rare ($0.17\%$) fraud instances.
* **Verdict:** Handled dataset complexity beautifully, maximizing classification confidence.
* **Validation Results:**
  * **Precision:** `0.81`
  * **Recall:** `0.87`
  * **F1-Score:** `0.84706`
* **Action Taken:** Serialized and saved into a production artifact (`models/fraud_model.pkl`).

### 3. Final Verification: Test Dataset (Unseen Data)
* **Verdict:** Evaluating the saved model against `test.csv` yielded highly consistent metrics. The trivial variance between validation and test scores explicitly **proves strong model generalization** and rules out overfitting.
* **Test Results (`test.csv`):**
  * **Precision:** `0.81`
  * **Recall:** `0.87`
  * **F1-Score:** `0.83582`

### 📈 Performance Comparison Matrix

| Model | Evaluation Phase | Precision | Recall | F1-Score | Operational Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Validation Data | `0.05` | `0.89` | `0.09195` | **Underfitting** (Too weak) |
| **Random Forest** | Validation Data | `0.81` | `0.87` | `0.84706` | **Optimal Performance** |
| **Random Forest (Saved `.pkl`)** | **Unseen Test Data** | **`0.81`** | **`0.87`** | **`0.83582`** | **Strong Generalization** |

---

## 🔄 End-to-End System Pipelines

The project architecture is split into three clean, repeatable workflows designed for production scalability:

### 🏭 1. Training Pipeline
* **Data Ingestion:** Imports raw historical training files (`train.csv`).
* **Feature Engineering:** Computes and introduces new custom transaction variables.
* **Data Cleaning:** Filters out duplicate records to prevent mathematical evaluation bias.
* **Target Splitting:** Divides features (`X_train`) away from target labels (`y_train`).
* **Data Preprocessing:** Executes a combined `.fit_transform()` to map numerical bounds, then serializes the state to `models/preprocessor.pkl`.
* **Model Training:** Trains the Random Forest ensemble and saves the model parameters.

### 🧪 2. Validation Pipeline
* **Data Ingestion:** Loads development validation files (`val.csv`).
* **Feature Engineering:** Generates the exact same suite of custom features.
* **Data Cleaning:** Passes the raw dataframe completely intact (retains duplicates to protect natural data evaluation balances).
* **Target Splitting:** Isolates features (`X_val`) from metrics (`y_val`).
* **Data Preprocessing:** Employs `.transform()` using the pre-saved parameters from the training step (no recalculations).
* **Model Evaluation:** Tracks accuracy metrics to allow manual tuning adjustments.

### 🚀 3. Testing Pipeline
* **Data Ingestion:** Fetches independent live or testing datasets (`test.csv`).
* **Feature Engineering:** Standardizes the shape by appending identical feature properties.
* **Target Splitting:** Extracts attributes (`X_test`) and evaluation keys (`y_test`).
* **Preprocessor Loading:** Reloads `preprocessor.pkl` to scale data formats via `.transform()`.
* **Prediction & Evaluation:** Loads `fraud_model.pkl`, **calculates final classifications**, and evaluates performance against ground truth.

---

## 🛣️ Future Roadmap
* [ ] **FastAPI Integration:** Wrap the prediction pipeline inside a lightweight, highly efficient FastAPI script to serve predictions over real-time HTTP requests.
* [ ] **Docker Containerization:** Package the code, dependencies, preprocessor artifacts, and trained models inside an isolated Docker container to ensure seamless deployment across cloud architectures.
