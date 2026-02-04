from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__, template_folder='../templates')

# Load models
baseline_model = joblib.load(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\baseline_model.pkl')
fair_model = joblib.load(r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\model\fair_model.pkl')

# Feature names (must match training data)
FEATURE_NAMES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Create feature array
        features = []
        for feature in FEATURE_NAMES:
            features.append(float(data.get(feature, 0)))
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Make predictions
        baseline_pred = baseline_model.predict(input_df)[0]
        baseline_proba = baseline_model.predict_proba(input_df)[0]
        
        fair_pred = fair_model.predict(input_df)[0]
        
        # Prepare response
        result = {
            'baseline_prediction': 'Default Risk' if baseline_pred == 1 else 'No Default Risk',
            'baseline_probability': f"{baseline_proba[1]*100:.2f}%",
            'fair_prediction': 'Default Risk' if fair_pred == 1 else 'No Default Risk',
            'gender': 'Male' if features[1] == 1 else 'Female',
            'decision': 'APPROVED' if fair_pred == 0 else 'REJECTED',
            'fairness_note': 'This prediction uses bias mitigation to ensure fair treatment across genders.'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
