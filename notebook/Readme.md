# Bias-Aware Credit Risk Scoring System

This project builds a machine learning-based credit risk model with fairness-aware techniques to detect and mitigate bias in loan approval decisions.

## Data Understanding
### UCI Credit Default Dataset
#### 1. Dataset Overview

Source: UCI Machine Learning Repository – Credit Card Default Dataset
-Total records: 30,000 customers
-Total features: 25 columns
-Missing values: ❌ None (all columns have 30,000 non-null values)
-Data types: All columns are numerical (int64)
-Memory size: ~5.7 MB

#### 2. Target Variable (Label)

-Column name:
default payment next month
Meaning:
1 → Customer defaults on payment
0 → Customer does not default


#### 3. Sensitive Attributes (Bias-Related Features)

-These attributes can introduce social bias and must be handled carefully:
Column		Meaning				Notes
SEX		Gender (1 = Male, 2 = Female)	Sensitive attribute
AGE		Customer age			Used to form age groups
EDUCATION	Education level			Socio-economic indicator
MARRIAGE	Marital status			Sensitive personal attribute

These attributes will be evaluated for disparate impact and fairness metrics in later phases.

-Problem type: Binary Classification
-Objective: Predict whether a customer will default next month

#### 4. Feature Categories
-Identifier (to be dropped)
Column
-ID

-Reason: Unique identifier, no predictive value.
Financial / Numeric Features
Type					Columns
Credit limit				LIMIT_BAL
Bill amounts (last 6 months)		BILL_AMT1 – BILL_AMT6
Payment amounts (last 6 months) 	PAY_AMT1 – PAY_AMT6

Behavioral / Credit History Features
Column			Meaning
PAY_0 – PAY_6		Repayment status for past 6 months

-Values represent:

-1 → Paid on time
0 → Revolving credit
1+ → Delay in payment (higher = worse behavior)

-Categorical (Encoded as Integers)

Even though they are integers, these are categorical in nature:

-Column

SEX
EDUCATION
MARRIAGE

-These may require:

Value grouping
Cleaning invalid codes
Fairness evaluation

#### 5. Data Quality Observations

-No missing values
-Clean numeric format
-Categorical variables are encoded, not one-hot
-Sensitive attributes present → bias risk
-Target column name contains spaces (will rename later)

#### 6. Initial Modeling Notes

-Dataset is suitable for:
-Logistic Regression
-Tree-based models
-Bias-aware ML (Fairlearn / AIF360)
-ID must be removed before training
-Sensitive attributes must be tracked, not blindly removed

#### 7. Summary (One-Line)

-The UCI Credit Default dataset contains 30,000 customer records with financial, behavioral, and sensitive demographic attributes, making it suitable for credit risk prediction and fairness-aware machine learning analysis.


## PHASE 3— DATA PREPROCESSING
###STEP 5 — Clean the Data
#### 1. Data Cleaning

Checked for missing values across all columns
No missing values were found in the dataset
Duplicate records were identified and removed (none found)
The ID column was dropped as it has no predictive value

#### 2.Feature Encoding

The following categorical features were identified:

SEX
EDUCATION
MARRIAGE

These features were retained in encoded form for fairness analysis

#### 3. Feature Scaling

Continuous numerical features were normalized using StandardScaler
Scaled features include:
LIMIT_BAL, AGE
BILL_AMT1 to BILL_AMT6
PAY_AMT1 to PAY_AMT6
Scaling ensures better model convergence and performance

#### 4. Sensitive Feature and Target Definition

Target variable: default payment next month
Sensitive attribute: SEX
Sensitive attribute is preserved for bias and fairness evaluation

#### 5. Output

The cleaned and preprocessed dataset was saved as:
cleaned_credit_data.csv
Dataset was saved without index values to avoid introducing artificial features

#### 6. Outcome

The dataset is clean, normalized, and ready for model training and bias-aware analysis.



### PHASE 4 — BASELINE MODEL DEVELOPMENT
#### 1. Objective
 -To develop an initial credit risk prediction model without applying any fairness constraints, serving as a reference for bias evaluation.

#### 2. Things Done
-Selected Logistic Regression as the baseline model.
-Split dataset into 80% training and 20% testing using stratified sampling.
-Trained the model using cleaned credit data.
-Evaluated performance using standard classification metrics.
-Saved the trained model and test predictions for further analysis.

#### 3. Performance Metrics
-Accuracy: 0.8080
-Precision: 0.6890
-Recall: 0.2404
-ROC-AUC: 0.7075

#### 4. Observation
-The model achieves reasonable accuracy but exhibits low recall for default cases.
-The performance imbalance between default and non-default classes indicates potential bias.
-These results justify the need for fairness evaluation and bias mitigation.

### PHASE 5 — BIAS DETECTION & FAIRNESS ANALYSIS
#### 1. Objective
-To identify and quantify bias present in the baseline credit risk model.

#### 2. Things Done
-Selected Gender (SEX) as the sensitive attribute.
-Evaluated fairness using the following metrics:
-Demographic Parity Difference
-Equalized Odds Difference
-Disparate Impact Ratio
-Compared approval rates across demographic groups.

#### 3. Fairness Metrics
-Demographic Parity Difference: 0.0247
-Equalized Odds Difference: 0.0313
-Disparate Impact Ratio: 0.731

#### 4. Observation
-The baseline model shows unequal approval rates between male and female applicants.
-The disparate impact ratio falls outside the acceptable fairness range.
-The model demonstrates demographic bias, requiring corrective measures.

### PHASE 6 — BIAS MITIGATION USING FAIR ML
#### 1. Objective
-To reduce demographic bias while maintaining acceptable predictive performance.

#### 2. Things Done
-Applied Exponentiated Gradient Reduction technique.
-Used Demographic Parity as the fairness constraint.
-Retrained the credit risk model under fairness constraints.
-Generated a fairness-aware credit scoring model.

#### 3. Observation
-Bias mitigation significantly reduced demographic disparities.
-The fairness-aware model maintained comparable accuracy to the baseline model.
-Trade-offs between fairness and performance were minimal.

### PHASE 7 — MODEL COMPARISON & EVALUATION
#### 1. Objective
-To compare baseline and fairness-aware models in terms of performance and fairness.

#### 2.Comparison Table
| Metric                        | Baseline Model | Fair Model |
| ----------------------------- | -------------- | ---------- |
| Accuracy                      | 0.8080         | 0.8083     |
| Precision                     | 0.6890         | 0.6911     |
| Recall                        | 0.2404         | 0.2411     |
| Demographic Parity Difference | 0.0247         | 0.0081     |
| Equalized Odds Difference     | 0.0313         | 0.0070     |

#### 3. Observation
-Fairness metrics improved significantly after mitigation.
-Predictive performance remained stable.
-The results demonstrate the effectiveness of fairness-aware ML techniques.

### PHASE 8 — EXPLAINABLE AI
#### 1. Objective
-To enhance transparency and interpretability of credit decisions.

#### 2. Things Done
-Applied SHAP for model interpretability.
-Identified globally important features influencing predictions.
-Explained individual credit decisions using SHAP values.

#### 3. Observation
-Financial attributes such as payment history and credit limit strongly influence predictions.
-Explainability improves trust and accountability in automated credit systems.

### PHASE 9 — DEPLOYMENT
#### 1. Objective
-To demonstrate real-time usability of the bias-aware model.

#### 2. Things Done
-Developed a simple Flask-based web application.
-Enabled user input for applicant details.
-Displayed credit decision results.

#### 3. Observation
-The system successfully generates real-time predictions.
-Demonstrates practical application of fairness-aware ML.

### PHASE 10 — CONCLUSION & FUTURE SCOPE
#### Conclusion
-The project demonstrates that fairness-aware machine learning can be effectively integrated into credit risk assessment systems to reduce bias while maintaining predictive performance.