import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib

# Load cleaned data
df = pd.read_csv(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\cleaned_credit_data.csv')

# Define features and target
target = 'default payment next month'
X = df.drop(columns=[target])
y = df[target]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 50)
print("TRAINING BASELINE MODEL")
print("=" * 50)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Step 8: Evaluate Performance
print("\n📊 MODEL PERFORMANCE:")
print("-" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# Save the model
joblib.dump(model, r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\baseline_model.pkl')
print("\n✅ Model saved as 'baseline_model.pkl'")

# Save test data for bias analysis
test_data = X_test.copy()
test_data[target] = y_test
test_data['predictions'] = y_pred
test_data.to_csv(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\test_predictions.csv', index=False)
print("✅ Test predictions saved!")
