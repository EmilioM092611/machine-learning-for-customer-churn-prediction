![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Customer Churn Prediction

## Project Overview

This project develops a machine learning classification model to predict customer churn using the IBM Telco Customer Churn dataset.

The objective is to identify customers who are more likely to discontinue their service and generate actionable insights that could support customer retention strategies.

The project follows a complete Data Science workflow, including data understanding, data cleaning, exploratory data analysis, feature engineering, model training, evaluation, hyperparameter tuning, and model interpretation.

---

## Business Problem

Customer churn is a significant challenge for subscription-based businesses.

When customers leave, companies lose recurring revenue and may need to invest additional resources in acquiring new customers.

A predictive churn model can help identify customers who are at higher risk of leaving, allowing the business to prioritize retention efforts before churn occurs.

### Objective

Predict whether a customer is likely to churn based on their demographic characteristics, account information, subscribed services, contract type, and billing information.

### Target Variable

The target variable is `Churn`.

- `Yes`: Customer churned
- `No`: Customer remained

---

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains 7,043 customer records and 21 variables covering demographic information, services, contracts, and billing characteristics.

### Main Feature Groups

- Customer demographics
- Customer tenure
- Phone services
- Internet services
- Security and support services
- Contract information
- Payment information
- Monthly and total charges

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook
- Git
- GitHub

---

## Project Structure

```text
machine-learning-for-customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   ├──WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├──customer_churn_clean.csv
│
├── images/
│   ├──Boxplots/
│   ├──Categorical Plots/
│   ├──Histograms/
│   ├──Matrices/
│
├── models/
│
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_model_interpretation.ipynb
│   └── 09_conclusions.ipynb
│
├── .gitignore
├── README.md
├── requirements.txt
└── Instrucciones.txt
```

---

## Business Questions

The project addresses the following business questions:

1. Which customer characteristics are most strongly associated with churn?
2. Which characteristics are associated with customer retention?
3. Which customer profiles should the company prioritize for retention campaigns?
4. Are contract type and tenure important predictors of churn?
5. Which services or customer characteristics appear to be associated with higher churn risk?
6. How can these findings support customer retention strategies?

The analysis indicates that customer tenure and contract type are particularly relevant predictors of churn. These findings can support proactive retention strategies by helping the company prioritize customers with higher predicted churn risk.

## Methodology

The project was developed through the following stages:

### 1. Data Understanding

The dataset structure, dimensions, data types, numerical variables, categorical variables, missing values, and duplicate records were analyzed.

The initial analysis confirmed that the dataset contains both numerical and categorical features, requiring different preprocessing techniques before model training.

### 2. Data Cleaning

Data quality issues were identified and addressed.

The main issue was the `TotalCharges` variable, which was initially stored as a string despite representing a numerical quantity.

The invalid values were identified and converted to missing values before being handled appropriately.

No duplicate records were found in the dataset.

### 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns associated with churn.

The analysis included:

- Target variable distribution
- Numerical variable distributions
- Categorical variable distributions
- Churn rates across customer segments
- Relationships between customer characteristics and churn
- Correlation analysis
- Outlier inspection

The target variable was moderately imbalanced, with fewer customers experiencing churn than remaining with the company.

### 4. Feature Engineering

Feature preprocessing was performed using Scikit-Learn pipelines.

Categorical variables were transformed using One-Hot Encoding.

Numerical variables were standardized using `StandardScaler`.

This preprocessing approach allows the transformations to be applied consistently during training and prediction while reducing the risk of data leakage.

### 5. Model Training

Three classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

These models were selected to compare different approaches to binary classification, from a linear model to tree-based algorithms.

### 6. Model Evaluation

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

Because customer churn is a classification problem with an imbalanced target variable, multiple evaluation metrics were considered rather than relying exclusively on accuracy.

### 7. Hyperparameter Tuning

`GridSearchCV` with cross-validation was used to optimize selected hyperparameters for Logistic Regression and Random Forest.

The tuned Logistic Regression model achieved the best overall balance between predictive performance and interpretability.

### 8. Model Interpretation

The final Logistic Regression model was interpreted using its coefficients.

Permutation importance was also analyzed to identify the features that contributed most strongly to the model's predictive performance.

---

## Model Performance

The tuned Logistic Regression model was selected as the final model.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 79.67% |
| Precision | 63.17% |
| Recall    | 56.42% |
| F1-Score  | 59.60% |
| ROC-AUC   | 83.49% |

During hyperparameter tuning, the model achieved a cross-validation ROC-AUC of **84.62%**.

The ROC-AUC score indicates that the model has a good ability to distinguish between customers who churn and customers who remain.

---

## Visualizations

The following visuals summarize the exploratory analysis, model comparison, and feature interpretation results discussed throughout this project.

### Exploratory Data Analysis

<img src="images/Churn Distribution.png" width="800">

### Model Comparison

<img src="images/Model Comparison - Accuracy.png" width="800">

### Feature Importance

<img src="images/Top Features by Permutation Importance.png" width="800">

<img src="images/Top Logistic Regression Coefficients.png" width="800">

### Confusion Matrix

<img src="images/confusion_matrix.png" width="500">

---

## Key Findings

The analysis identified several customer characteristics associated with churn.

The most relevant features included:

- Customer tenure
- Contract type
- Internet service type
- Monthly charges
- Total charges
- Payment method

### Customer Tenure

`tenure` was the most influential feature according to the permutation importance analysis.

Customers with shorter relationships with the company showed a higher tendency to churn, while longer-tenured customers were generally more likely to remain.

### Contract Type

Customers with month-to-month contracts showed a stronger association with churn.

Longer-term contracts, particularly two-year contracts, were associated with lower churn risk.

### Internet Service

Internet service type was also an important predictive feature.

Customers using fiber optic service showed a stronger association with churn compared with some other service groups.

### Billing and Charges

Monthly charges and total charges were also relevant to the model.

Higher monthly charges were associated with greater churn risk in the model, while total charges also contributed to the prediction.

These relationships should be interpreted as predictive associations rather than causal effects.

---

## Model Interpretation

The final Logistic Regression model provided interpretable coefficients that indicate how different features influence the predicted probability of churn.

Positive coefficients are associated with higher predicted churn probability, while negative coefficients are associated with lower predicted churn probability, assuming the encoded feature representation is held constant.

The strongest positive coefficients included:

- `TotalCharges`
- `Contract_Month-to-month`
- `InternetService_Fiber optic`
- `StreamingTV_Yes`
- `PaymentMethod_Electronic check`

The strongest negative coefficients included:

- `tenure`
- `Contract_Two year`
- `InternetService_DSL`
- `MonthlyCharges`

Permutation importance identified `tenure` as the most influential feature at the original feature level.

Model interpretation should be understood as predictive association rather than evidence of causality.

---

## Business Recommendations

Based on the analysis, the company could:

1. Prioritize retention campaigns for newer customers.

2. Develop incentives that encourage customers to transition from month-to-month contracts to longer-term contracts.

3. Identify customers with high predicted churn probability and prioritize them for proactive retention campaigns.

4. Investigate service-related factors associated with higher churn rates, particularly internet service characteristics.

5. Monitor customers with high monthly charges and evaluate whether pricing, perceived value, or service experience may contribute to churn.

6. Use predictive modeling as a decision-support tool for customer retention strategies.

The model should be used as a prioritization tool rather than as a standalone decision-making system.

---

## Limitations

Several limitations should be considered when interpreting the results:

- The dataset represents a specific customer population and may not generalize to other companies or markets.
- The model identifies predictive associations rather than causal relationships.
- The dataset contains limited historical customer behavior.
- The dataset does not provide detailed temporal information about customer interactions.
- Business costs associated with false positives and false negatives were not incorporated into the model.
- Model performance may change when applied to new customer populations or future data.

---

## Future Improvements

Potential future improvements include:

- Testing Gradient Boosting, XGBoost, or LightGBM.
- Performing more extensive hyperparameter optimization.
- Optimizing the classification threshold based on business costs.
- Applying SHAP for more detailed model interpretation.
- Incorporating Customer Lifetime Value analysis.
- Incorporating historical customer behavior.
- Developing a model deployment API.
- Creating a customer churn prediction application.
- Monitoring model performance after deployment.
- Implementing model retraining strategies as new customer data becomes available.

---

## Project Outputs

The project produces:

- Cleaned and validated customer data
- Exploratory data analysis
- Feature engineering pipeline
- Multiple trained classification models
- Model performance comparison
- Hyperparameter tuning
- Final churn prediction model
- Model interpretation
- Business recommendations
- Professional project documentation

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Navigate to the project directory

```bash
cd machine-learning-for-customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows

```powershell
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and execute them in the recommended order.

---

## Conclusion

This project demonstrates a complete machine learning workflow for a customer churn classification problem, from data understanding and exploratory analysis to model development, evaluation, interpretation, and business recommendations.

The final Logistic Regression model achieved an ROC-AUC of **83.49%** on the test set, providing a useful predictive baseline for identifying customers at higher risk of churn.

Beyond model performance, the project demonstrates how machine learning can be connected to a real business problem and transformed into actionable insights for customer retention.

---

## Author

Emilio Márquez Morales

Data Analyst jr. | Software Engineer

Data Portfolio Project
