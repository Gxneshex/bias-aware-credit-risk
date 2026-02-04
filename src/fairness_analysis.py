import pandas as pd
import joblib
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio
)

# Load test data with predictions
test_data = pd.read_csv(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\test_predictions.csv')

# Extract variables
sensitive_feature = test_data['SEX']  # 1=male, 2=female
y_true = test_data['default payment next month']
y_pred = test_data['predictions']

print("=" * 60)
print("BIAS DETECTION ANALYSIS")
print("=" * 60)

# Step 10: Compute Bias Metrics
print("\n📊 FAIRNESS METRICS:")
print("-" * 60)

# Demographic Parity Difference
dp_diff = demographic_parity_difference(
    y_true, y_pred, sensitive_features=sensitive_feature
)
print(f"Demographic Parity Difference: {dp_diff:.4f}")
print("  → Measures difference in positive prediction rates")
print("  → Ideal value: 0 (perfectly fair)")

# Equalized Odds Difference
eo_diff = equalized_odds_difference(
    y_true, y_pred, sensitive_features=sensitive_feature
)
print(f"\nEqualized Odds Difference: {eo_diff:.4f}")
print("  → Measures difference in error rates")
print("  → Ideal value: 0 (perfectly fair)")

# Disparate Impact (Demographic Parity Ratio)
di_ratio = demographic_parity_ratio(
    y_true, y_pred, sensitive_features=sensitive_feature
)
print(f"\nDisparate Impact Ratio: {di_ratio:.4f}")
print("  → Ratio of positive prediction rates")
print("  → Ideal value: 1.0 (perfectly fair)")
print("  → Acceptable range: 0.8 to 1.25")

# Step 11: Create Approval Rate Table
print("\n" + "=" * 60)
print("APPROVAL RATES BY GENDER")
print("=" * 60)

approval_by_gender = test_data.groupby('SEX')['predictions'].agg([
    ('Total', 'count'),
    ('Approved (Predicted Default)', 'sum'),
    ('Approval Rate', 'mean')
])

approval_by_gender.index = ['Male (1)', 'Female (2)']
print(approval_by_gender)

# Calculate difference
male_rate = approval_by_gender.loc['Male (1)', 'Approval Rate']
female_rate = approval_by_gender.loc['Female (2)', 'Approval Rate']
difference = abs(male_rate - female_rate)

print(f"\n📈 Approval Rate Difference: {difference:.4f} ({difference*100:.2f}%)")

# Save results
results = {
    'Metric': [
        'Demographic Parity Difference',
        'Equalized Odds Difference',
        'Disparate Impact Ratio',
        'Male Approval Rate',
        'Female Approval Rate',
        'Approval Rate Difference'
    ],
    'Value': [
        dp_diff,
        eo_diff,
        di_ratio,
        male_rate,
        female_rate,
        difference
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv(
    r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\bias_metrics.csv',
    index=False
)
print("\n✅ Bias metrics saved to 'bias_metrics.csv'")

# Interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
if abs(dp_diff) > 0.1:
    print("⚠️  BIAS DETECTED: Significant demographic parity difference")
else:
    print("✅ Low demographic parity difference")

if di_ratio < 0.8 or di_ratio > 1.25:
    print("⚠️  BIAS DETECTED: Disparate impact outside acceptable range")
else:
    print("✅ Disparate impact within acceptable range")
