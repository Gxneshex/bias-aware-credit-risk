Python 3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> 
= RESTART: C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\src\train_model.py
==================================================
TRAINING BASELINE MODEL
==================================================

📊 MODEL PERFORMANCE:
--------------------------------------------------
Accuracy:  0.8080
Precision: 0.6890
Recall:    0.2404
ROC-AUC:   0.7075

📋 Classification Report:
              precision    recall  f1-score   support

  No Default       0.82      0.97      0.89      4673
     Default       0.69      0.24      0.36      1327

    accuracy                           0.81      6000
   macro avg       0.75      0.60      0.62      6000
weighted avg       0.79      0.81      0.77      6000


✅ Model saved as 'baseline_model.pkl'
✅ Test predictions saved!
>>> 
= RESTART: C:/Users/rsury/OneDrive/Desktop/bias-aware-credit-risk/src/fairness_analysis.py
============================================================
BIAS DETECTION ANALYSIS
============================================================

📊 FAIRNESS METRICS:
------------------------------------------------------------
Demographic Parity Difference: 0.0247
  → Measures difference in positive prediction rates
  → Ideal value: 0 (perfectly fair)

Equalized Odds Difference: 0.0313
  → Measures difference in error rates
  → Ideal value: 0 (perfectly fair)

Disparate Impact Ratio: 0.7310
  → Ratio of positive prediction rates
  → Ideal value: 1.0 (perfectly fair)
  → Acceptable range: 0.8 to 1.25

============================================================
APPROVAL RATES BY GENDER
============================================================
            Total  Approved (Predicted Default)  Approval Rate
Male (1)     2402                           221       0.092007
Female (2)   3598                           242       0.067260

📈 Approval Rate Difference: 0.0247 (2.47%)

✅ Bias metrics saved to 'bias_metrics.csv'

============================================================
INTERPRETATION
============================================================
✅ Low demographic parity difference
⚠️  BIAS DETECTED: Disparate impact outside acceptable range

= RESTART: C:/Users/rsury/OneDrive/Desktop/bias-aware-credit-risk/src/bias_mitigation.py
============================================================
BIAS MITIGATION - TRAINING FAIR MODEL
============================================================

🔄 Training fair model with demographic parity constraint...
✅ Fair model trained!

============================================================
MODEL COMPARISON
============================================================
                       Metric  Baseline Model  Fair Model
                     Accuracy        0.808000    0.808333
                    Precision        0.688985    0.691145
                       Recall        0.240392    0.241145
Demographic Parity Difference        0.024747    0.008085
    Equalized Odds Difference        0.031313    0.007049

✅ Fair model saved as 'fair_model.pkl'
✅ Comparison saved to 'model_comparison.csv'

============================================================
APPROVAL RATES COMPARISON
============================================================

Baseline Model:
  Male (1):   0.0920
  Female (2): 0.0673
  Difference: 0.0247

Fair Model:
  Male (1):   0.0820
  Female (2): 0.0739
  Difference: 0.0081

= RESTART: C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\src\preprocess.py
============================================================
DATA PREPROCESSING
============================================================

📊 Original dataset shape: (30000, 25)
Missing values: 0
Duplicates before removal: 0
Duplicates after removal: 0

✅ Final dataset shape: (30000, 24)
✅ Features normalized: 14

✅ Cleaned data saved to: cleaned_credit_data.csv

= RESTART: C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\src\explain.py
============================================================
MODEL EXPLAINABILITY WITH SHAP
============================================================

🔍 Generating SHAP explanations...

📊 Analyzing Baseline Model...
✅ Saved: shap_summary_baseline.png
✅ Saved: shap_importance_baseline.png

📊 Analyzing Fair Model...

PermutationExplainer explainer:  72%|███████▏  | 72/100 [00:00<?, ?it/s]
PermutationExplainer explainer:  76%|███████▌  | 76/100 [00:10<00:00, 35.44it/s]
PermutationExplainer explainer:  80%|████████  | 80/100 [00:10<00:00, 28.98it/s]
PermutationExplainer explainer:  83%|████████▎ | 83/100 [00:10<00:00, 27.29it/s]
PermutationExplainer explainer:  86%|████████▌ | 86/100 [00:10<00:00, 24.37it/s]
PermutationExplainer explainer:  89%|████████▉ | 89/100 [00:10<00:00, 22.17it/s]
PermutationExplainer explainer:  92%|█████████▏| 92/100 [00:10<00:00, 21.10it/s]
PermutationExplainer explainer:  95%|█████████▌| 95/100 [00:11<00:00, 20.67it/s]
PermutationExplainer explainer:  98%|█████████▊| 98/100 [00:11<00:00, 19.68it/s]
PermutationExplainer explainer: 100%|██████████| 100/100 [00:11<00:00, 18.23it/s]
PermutationExplainer explainer: 101it [00:11,  2.51it/s]                         

Warning (from warnings module):
  File "C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\src\explain.py", line 74
    shap.summary_plot(shap_values_fair, X_sample, show=False)
FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
✅ Saved: shap_summary_fair.png
✅ Saved: shap_importance_fair.png

🔍 Creating individual prediction explanation...
✅ Saved: shap_individual_example.png

📈 Comparing feature importance...

Top 10 Most Important Features (Baseline Model):
  Feature  Importance
    PAY_0    0.454532
BILL_AMT1    0.220874
LIMIT_BAL    0.103216
 MARRIAGE    0.071808
    PAY_3    0.068905
EDUCATION    0.068568
    PAY_2    0.066130
      AGE    0.062028
      SEX    0.056196
BILL_AMT2    0.048379

✅ Feature importance saved to: feature_importance.csv

============================================================
EXPLAINABILITY ANALYSIS COMPLETE
============================================================

📊 Generated files:
  • shap_summary_baseline.png
  • shap_importance_baseline.png
  • shap_summary_fair.png
  • shap_importance_fair.png
  • shap_individual_example.png
  • feature_importance.csv

== RESTART: C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\src\app.py ==
 * Serving Flask app 'app'
 * Debug mode: on
[31m[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.[0m
 * Running on http://127.0.0.1:5000
[33mPress CTRL+C to quit[0m
 * Restarting with stat

== RESTART: C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\src\app.py ==
 * Debugger is active!
 * Debugger PIN: 824-735-167
127.0.0.1 - - [02/Feb/2026 15:03:39] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [02/Feb/2026 15:03:39] "[33mGET /favicon.ico HTTP/1.1[0m" 404 -
127.0.0.1 - - [02/Feb/2026 15:03:55] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [02/Feb/2026 15:03:55] "[33mGET /favicon.ico HTTP/1.1[0m" 404 -
127.0.0.1 - - [02/Feb/2026 15:03:59] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [02/Feb/2026 15:04:00] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [02/Feb/2026 15:06:56] "GET / HTTP/1.1" 200 -
 * Detected change in 'C:\\Users\\rsury\\OneDrive\\Desktop\\bias-aware-credit-risk\\src\\app.py', reloading
Exception in thread Thread-1 (serve_forever):
Traceback (most recent call last):
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.2544.0_x64__qbz5n2kfra8p0\Lib\threading.py", line 1043, in _bootstrap_inner
    self.run()
    ~~~~~~~~^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.2544.0_x64__qbz5n2kfra8p0\Lib\threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rsury\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\werkzeug\serving.py", line 819, in serve_forever
