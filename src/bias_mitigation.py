import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Load cleaned data
df = pd.read_csv(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\cleaned_credit_data.csv')

target = 'default payment next month'
sensitive_feature_col = 'SEX'

X = df.drop(columns=[target])
y = df[target]
sensitive_feature = df[sensitive_feature_col]

# Split data
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("BIAS MITIGATION - TRAINING FAIR MODEL")
print("=" * 60)

# Step 12: Apply Bias Mitigation using Fairlearn
base_estimator = LogisticRegression(max_iter=1000, random_state=42)

# Use Exponentiated Gradient with Demographic Parity constraint
mitigator = ExponentiatedGradient(
    estimator=base_estimator,
    constraints=DemographicParity()
)

print("\n🔄 Training fair model with demographic parity constraint...")
mitigator.fit(X_train, y_train, sensitive_features=A_train)
print("✅ Fair model trained!")

# Step 13: Predictions from fair model
y_pred_fair = mitigator.predict(X_test)

# Load baseline model for comparison
baseline_model = joblib.load(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\baseline_model.pkl')
y_pred_baseline = baseline_model.predict(X_test)

# Step 14: Compare Models
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall',
        'Demographic Parity Difference',
        'Equalized Odds Difference'
    ],
    'Baseline Model': [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_baseline),
        demographic_parity_difference(y_test, y_pred_baseline, sensitive_features=A_test),
        equalized_odds_difference(y_test, y_pred_baseline, sensitive_features=A_test)
    ],
    'Fair Model': [
        accuracy_score(y_test, y_pred_fair),
        precision_score(y_test, y_pred_fair),
        recall_score(y_test, y_pred_fair),
        demographic_parity_difference(y_test, y_pred_fair, sensitive_features=A_test),
        equalized_odds_difference(y_test, y_pred_fair, sensitive_features=A_test)
    ]
})

print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\model_comparison.csv',
    index=False
)

# Save fair model
joblib.dump(mitigator, r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\fair_model.pkl')
print("\n✅ Fair model saved as 'fair_model.pkl'")
print("✅ Comparison saved to 'model_comparison.csv'")

# Approval rates comparison
print("\n" + "=" * 60)
print("APPROVAL RATES COMPARISON")
print("=" * 60)

test_comparison = pd.DataFrame({
    'SEX': A_test,
    'Baseline_Pred': y_pred_baseline,
    'Fair_Pred': y_pred_fair
})

baseline_rates = test_comparison.groupby('SEX')['Baseline_Pred'].mean()
fair_rates = test_comparison.groupby('SEX')['Fair_Pred'].mean()

print("\nBaseline Model:")
print(f"  Male (1):   {baseline_rates[1]:.4f}")
print(f"  Female (2): {baseline_rates[2]:.4f}")
print(f"  Difference: {abs(baseline_rates[1] - baseline_rates[2]):.4f}")

print("\nFair Model:")
print(f"  Male (1):   {fair_rates[1]:.4f}")
print(f"  Female (2): {fair_rates[2]:.4f}")
print(f"  Difference: {abs(fair_rates[1] - fair_rates[2]):.4f}")
