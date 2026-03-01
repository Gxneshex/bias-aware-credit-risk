import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def mitigate_bias():
    """Apply bias mitigation and train fair model"""
    
    print("=" * 60)
    print("BIAS MITIGATION - TRAINING FAIR MODEL")
    print("=" * 60)
    
    # Load cleaned data using Config
    print(f"\n📂 Loading data from: {Config.CLEAN_DATA}")
    df = pd.read_csv(Config.CLEAN_DATA)
    
    # Prepare features
    X = df.drop(columns=[Config.TARGET_COLUMN])
    y = df[Config.TARGET_COLUMN]
    sensitive_feature = df[Config.SENSITIVE_ATTRIBUTE]
    
    # Split data using Config parameters
    print(f"🔀 Splitting data (test_size={Config.TEST_SIZE})...")
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_feature, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, 
        stratify=y
    )
    
    # Apply bias mitigation
    print("\n🤖 Applying Exponentiated Gradient with Demographic Parity...")
    base_estimator = LogisticRegression(
        max_iter=Config.LOGISTIC_MAX_ITER, 
        random_state=Config.LOGISTIC_RANDOM_STATE
    )
    
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity()
    )
    
    print("🔄 Training fair model...")
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    print("✅ Fair model trained!")
    
    # Predictions
    y_pred_fair = mitigator.predict(X_test)
    
    # Load baseline model for comparison
    print(f"\n📂 Loading baseline model from: {Config.BASELINE_MODEL}")
    baseline_model = joblib.load(Config.BASELINE_MODEL)
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Compare models
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
    
    # Save comparison using Config
    print(f"\n💾 Saving comparison to: {Config.MODEL_COMPARISON}")
    comparison.to_csv(Config.MODEL_COMPARISON, index=False)
    print("✅ Comparison saved")
    
    # Save fair model using Config
    print(f"💾 Saving fair model to: {Config.FAIR_MODEL}")
    joblib.dump(mitigator, Config.FAIR_MODEL)
    print("✅ Fair model saved")
    
    # Approval rates comparison
    print("\n" + "=" * 60)
    print("APPROVAL RATES COMPARISON")
    print("=" * 60)
    
    test_comparison = pd.DataFrame({
        Config.SENSITIVE_ATTRIBUTE: A_test,
        'Baseline_Pred': y_pred_baseline,
        'Fair_Pred': y_pred_fair
    })
    
    baseline_rates = test_comparison.groupby(Config.SENSITIVE_ATTRIBUTE)['Baseline_Pred'].mean()
    fair_rates = test_comparison.groupby(Config.SENSITIVE_ATTRIBUTE)['Fair_Pred'].mean()
    
    print("\nBaseline Model:")
    print(f"  Male (1):   {baseline_rates[1]:.4f}")
    print(f"  Female (2): {baseline_rates[2]:.4f}")
    print(f"  Difference: {abs(baseline_rates[1] - baseline_rates[2]):.4f}")
    
    print("\nFair Model:")
    print(f"  Male (1):   {fair_rates[1]:.4f}")
    print(f"  Female (2): {fair_rates[2]:.4f}")
    print(f"  Difference: {abs(fair_rates[1] - fair_rates[2]):.4f}")
    
    # Calculate improvement
    baseline_gap = abs(baseline_rates[1] - baseline_rates[2])
    fair_gap = abs(fair_rates[1] - fair_rates[2])
    improvement = ((baseline_gap - fair_gap) / baseline_gap) * 100
    
    print(f"\n🎯 Bias Reduction: {improvement:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    mitigate_bias()
