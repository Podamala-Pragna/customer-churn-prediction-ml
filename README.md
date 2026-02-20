# Customer Churn Prediction in Telecom Industry

##  Project Overview

Customer churn happens when a customer stops using a company’s service. In the telecom industry, this directly affects revenue and customer retention strategies.  
The goal of this project is to predict whether a telecom customer is likely to churn based on their demographic details, account information, and service usage patterns.

Using the **Telco Customer Churn dataset**, we built and compared three machine learning models:
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  

Finally, we identified the best performing model and analyzed key factors influencing churn.

---

##  Problem Statement

The main objective is to predict customer churn in a telecom company using machine learning models.  
The model should help the company understand which customers are at higher risk of leaving and what factors contribute most to their decision.

---

##  Dataset Information

**Dataset Source:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- Total records: **7043**
- Total features: **21**
- Target variable: **Churn (Yes/No)**

The dataset includes:
- **Customer Demographics** – gender, senior citizen status, dependents, partner  
- **Account Information** – tenure, contract type, billing method, payment method  
- **Services Used** – phone, internet, streaming, online security, etc.  
- **Charges** – monthly and total charges  

---

##  Data Preprocessing

### 1. Loading and Inspection
The dataset was loaded successfully from `/content/WA_Fn-UseC_-Telco-Customer-Churn.csv`  
Shape of dataset: **(7043, 21)**  
We inspected data types, null values, and statistical summaries.

### 2. Handling Missing Values
- Found **11 missing or blank values** in the `TotalCharges` column (stored as text).  
- Converted to numeric and replaced missing entries with the **median** value.  
- Confirmed that there were **no missing values** remaining.

### 3. Removing Irrelevant Columns
- Dropped the `customerID` column since it is only an identifier and doesn’t contribute to prediction.

### 4. Data Cleaning Summary
After cleaning, the dataset contained:
- 20 meaningful features  
- 7043 complete rows  
- No null or inconsistent data  

---

##  Exploratory Data Analysis (EDA)

To understand relationships between features and churn, we created several visualizations using **Seaborn** and **Matplotlib**:

1. **Gender vs Churn** – Churn rates were quite similar across genders.  
2. **Senior Citizen vs Churn** – Senior citizens showed a higher churn percentage.  
3. **Monthly Charges vs Churn** – Customers with higher monthly charges were more likely to leave.  
4. **Contract Type vs Churn** – Month-to-month contracts had the highest churn rate.  
5. **Payment Method vs Churn** – Electronic check payments were associated with more churn.  
6. **Tenure Distribution by Churn** – Customers with shorter tenures were more likely to leave.  
7. **Correlation Heatmap** – Showed positive correlation between monthly charges and churn, and negative correlation between tenure and churn.

---

##  Feature Engineering and Encoding

- Created a new `tenure_group` feature to categorize customers by tenure range (0–12, 13–24, 25–36, 37–48, 49–60, 60+ months).
- Categorical variables were **encoded** using one-hot encoding.  
- Numerical features were **scaled** using `StandardScaler` for consistent magnitude.  
- The final dataset was split into **80% training** and **20% testing** sets using `train_test_split`.

---

##  Model Building

We implemented and trained the following models:

1. **Logistic Regression** – Baseline model for binary classification  
2. **Random Forest Classifier** – Ensemble-based model for handling complex relationships  
3. **Gradient Boosting Classifier** – Boosting algorithm for better accuracy with iterative learning  

Each model was trained on the same training data for fair comparison.

---

##  Model Evaluation

We evaluated all models using **Accuracy**, **Confusion Matrix**, and **Classification Report**.

|     Model            | Accuracy   |
|----------------------|------------|             
| Logistic Regression  | **0.8041** |
| Random Forest        | 0.7949     |
| Gradient Boosting    | **0.8041** |

 **Best Performing Model:** Logistic Regression (Accuracy = 80.41%)

**Reasoning:**
- Logistic Regression performed the best with balanced precision and recall.
- Gradient Boosting gave similar results but was more complex and slower to train.
- Random Forest performed slightly lower, around 79.49%.

---

##  Insights from Analysis

- Customers with **month-to-month contracts**, **higher monthly charges**, or **shorter tenure** were more likely to churn.  
- **Senior citizens** and customers paying via **electronic check** were at higher risk.  
- Customers with **long-term contracts** and **automatic payments** had lower churn rates.

---

##  Conclusion

This project successfully predicts telecom customer churn using real-world data.  
Among the three models tested, **Logistic Regression** achieved the best accuracy of **80.41%**, making it both effective and interpretable.  

By identifying key factors influencing churn, telecom providers can take targeted steps — such as offering loyalty discounts or improving service quality — to retain valuable customers.

---

##  How to Run the Notebook

1. Open the notebook in **Google Colab**.  
2. Upload the dataset file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`.  
3. Run each cell in sequence.  
4. The notebook will automatically generate all visualizations and model comparison outputs.  

*(Since this was done in Colab, no separate `requirements.txt` file is needed — all required libraries are pre-installed.)*

---

##  Libraries Used

- **pandas** – Data loading and manipulation  
- **numpy** – Numerical computations  
- **matplotlib** – Visualization  
- **seaborn** – Advanced plotting and EDA  
- **scikit-learn** – Machine learning model building and evaluation  

---

##  Future Enhancements

- Perform **hyperparameter tuning** to further improve accuracy.  
- Explore advanced models like **XGBoost** or **LightGBM**.  
- Build a **web interface or dashboard** for interactive churn prediction.  
- Incorporate **customer feedback data** for better prediction context.

---

### Submitted as part of the **Machine Learning (UE23CS352A)** mini project.
