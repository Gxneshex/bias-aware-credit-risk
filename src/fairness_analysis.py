import pandas as pd
import joblib
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio
)
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def measure_bias():
    """Measure algorithmic bias in baseline model"""
    
    print("=" * 60)
    print("BIAS DETECTION ANALYSIS")
    print("=" * 60)
    
    # Load test data with predictions using Config
    print(f"\n📂 Loading predictions from: {Config.TEST_PREDICTIONS}")
    test_data = pd.read_csv(Config.TEST_PREDICTIONS)
    
    # Extract variables using Config
    sensitive_feature = test_data[Config.SENSITIVE_ATTRIBUTE]  # Gender
    y_true = test_data[Config.TARGET_COLUMN]
    y_pred = test_data['predictions']
    
    print(f"📊 Analyzing {len(test_data)} predictions")
    print(f"🔍 Sensitive attribute: {Config.SENSITIVE_ATTRIBUTE}")
    
    # Compute bias metrics
    print("\n" + "=" * 60)
    print("📊 FAIRNESS METRICS")
    print("=" * 60)
    
    # Demographic Parity Difference
    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
    print("  → Measures difference in positive prediction rates")
    print("  → Ideal value: 0 (perfectly fair)")
    
    # Equalized Odds Difference
    eo_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    print(f"\nEqualized Odds Difference: {eo_diff:.4f}")
    print("  → Measures difference in error rates")
    print("  → Ideal value: 0 (perfectly fair)")
    
    # Disparate Impact Ratio
    di_ratio = demographic_parity_ratio(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    print(f"\nDisparate Impact Ratio: {di_ratio:.4f}")
    print("  → Ratio of positive prediction rates")
    print("  → Ideal value: 1.0 (perfectly fair)")
    print(f"  → Acceptable range: {Config.DISPARATE_IMPACT_MIN} to {Config.DISPARATE_IMPACT_MAX}")
    
    # Approval rates by gender
    print("\n" + "=" * 60)
    print("APPROVAL RATES BY GENDER")
    print("=" * 60)
    
    approval_by_gender = test_data.groupby(Config.SENSITIVE_ATTRIBUTE)['predictions'].agg([
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
    
    # Save results using Config
    print(f"\n💾 Saving bias metrics to: {Config.BIAS_METRICS}")
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
    results_df.to_csv(Config.BIAS_METRICS, index=False)
    print("✅ Bias metrics saved")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if abs(dp_diff) > Config.DEMOGRAPHIC_PARITY_THRESHOLD:
        print("⚠️  BIAS DETECTED: Significant demographic parity difference")
    else:
        print("✅ Low demographic parity difference")
    
    if di_ratio < Config.DISPARATE_IMPACT_MIN or di_ratio > Config.DISPARATE_IMPACT_MAX:
        print("⚠️  BIAS DETECTED: Disparate impact outside acceptable range")
    else:
        print("✅ Disparate impact within acceptable range")
    
    print("=" * 60)

if __name__ == "__main__":
    measure_bias()
