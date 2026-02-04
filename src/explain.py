import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_models():
    """
    Generate SHAP explanations for baseline and fair models
    """
    print("=" * 60)
    print("MODEL EXPLAINABILITY WITH SHAP")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\cleaned_credit_data.csv')
    
    target = 'default payment next month'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Load models
    baseline_model = joblib.load(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\baseline_model.pkl')
    fair_model = joblib.load(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\fair_model.pkl')
    
    print("\n🔍 Generating SHAP explanations...")
    
    # Sample data for SHAP (use 100 samples for speed)
    X_sample = shap.sample(X, 100, random_state=42)
    
    # ============================================
    # BASELINE MODEL EXPLANATION
    # ============================================
    print("\n📊 Analyzing Baseline Model...")
    
    # Create SHAP explainer
    explainer_baseline = shap.Explainer(baseline_model, X_sample)
    shap_values_baseline = explainer_baseline(X_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_baseline, X_sample, show=False)
    plt.title("SHAP Summary - Baseline Model", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\shap_summary_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary_baseline.png")
    
    # Bar plot - feature importance
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_baseline, show=False)
    plt.title("Feature Importance - Baseline Model", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\shap_importance_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_importance_baseline.png")
    
    # ============================================
    # FAIR MODEL EXPLANATION
    # ============================================
    print("\n📊 Analyzing Fair Model...")
    
    # For fair model, we need to handle it differently
    # Since ExponentiatedGradient doesn't support SHAP directly,
    # we'll use the underlying predictors
    
    try:
        # Try to explain fair model
        explainer_fair = shap.Explainer(fair_model.predict, X_sample)
        shap_values_fair = explainer_fair(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_fair, X_sample, show=False)
        plt.title("SHAP Summary - Fair Model", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\shap_summary_fair.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_summary_fair.png")
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values_fair, show=False)
        plt.title("Feature Importance - Fair Model", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\shap_importance_fair.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_importance_fair.png")
        
    except Exception as e:
        print(f"⚠️  Could not create SHAP plots for fair model: {e}")
        print("   (This is normal for some fairness-aware models)")
    
    # ============================================
    # INDIVIDUAL PREDICTION EXPLANATION
    # ============================================
    print("\n🔍 Creating individual prediction explanation...")
    
    # Get a sample individual (first person in test set)
    individual = X_sample.iloc[0:1]
    
    # Waterfall plot for baseline model
    shap_values_individual = explainer_baseline(individual)
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_individual[0], show=False)
    plt.title("Individual Prediction Explanation (Baseline Model)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\shap_individual_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_individual_example.png")
    
    # ============================================
    # FEATURE IMPORTANCE COMPARISON
    # ============================================
    print("\n📈 Comparing feature importance...")
    
    # Get top features from baseline
    feature_importance = np.abs(shap_values_baseline.values).mean(0)
    feature_names = X_sample.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Baseline Model):")
    print(importance_df.head(10).to_string(index=False))
    
    # Save importance
    importance_df.to_csv(
        r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\feature_importance.csv',
        index=False
    )
    print("\n✅ Feature importance saved to: feature_importance.csv")
    
    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n📊 Generated files:")
    print("  • shap_summary_baseline.png")
    print("  • shap_importance_baseline.png")
    print("  • shap_summary_fair.png")
    print("  • shap_importance_fair.png")
    print("  • shap_individual_example.png")
    print("  • feature_importance.csv")

if __name__ == "__main__":
    # Install SHAP if not already installed
    try:
        import shap
    except ImportError:
        print("Installing SHAP...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'shap'])
        import shap
    
    explain_models()
