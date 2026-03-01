import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def train_baseline():
    """Train baseline model without fairness constraints"""
    
    print("=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)
    
    # Load cleaned data using Config
    print(f"\n📂 Loading data from: {Config.CLEAN_DATA}")
    df = pd.read_csv(Config.CLEAN_DATA)
    
    # Define features and target using Config
    X = df.drop(columns=[Config.TARGET_COLUMN])
    y = df[Config.TARGET_COLUMN]
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Target distribution:")
    print(y.value_counts())
    
    # Split data using Config parameters
    print(f"\n🔀 Splitting data (test_size={Config.TEST_SIZE}, random_state={Config.RANDOM_STATE})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, 
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Logistic Regression using Config parameters
    print(f"\n🤖 Training Logistic Regression (max_iter={Config.LOGISTIC_MAX_ITER})...")
    model = LogisticRegression(
        max_iter=Config.LOGISTIC_MAX_ITER, 
        random_state=Config.LOGISTIC_RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete")
    
    # Predict on test set
    print("\n🔮 Generating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate performance
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    # Save the model using Config
    print(f"\n💾 Saving model to: {Config.BASELINE_MODEL}")
    joblib.dump(model, Config.BASELINE_MODEL)
    print("✅ Model saved successfully")
    
    # Save test data for bias analysis
    print(f"💾 Saving test predictions to: {Config.TEST_PREDICTIONS}")
    test_data = X_test.copy()
    test_data[Config.TARGET_COLUMN] = y_test
    test_data['predictions'] = y_pred
    test_data.to_csv(Config.TEST_PREDICTIONS, index=False)
    print("✅ Test predictions saved")
    
    print("\n" + "=" * 60)
    print("✅ BASELINE MODEL TRAINING COMPLETE")
    print("=" * 60)
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_baseline()
