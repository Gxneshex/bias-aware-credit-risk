"""
Configuration file for Bias-Aware Credit Risk Scoring System
Centralizes all paths, parameters, and settings
"""

import os

class Config:
    """Main configuration class"""
    
    # ============================================
    # DIRECTORY PATHS
    # ============================================
    BASE_DIR = r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # ============================================
    # DATA FILES
    # ============================================
    # Input data
    RAW_DATA = os.path.join(DATA_DIR, 'default of credit card clients.csv')
    CLEAN_DATA = os.path.join(DATA_DIR, 'cleaned_credit_data.csv')
    
    # Output data
    TEST_PREDICTIONS = os.path.join(DATA_DIR, 'test_predictions.csv')
    BIAS_METRICS = os.path.join(DATA_DIR, 'bias_metrics.csv')
    MODEL_COMPARISON = os.path.join(DATA_DIR, 'model_comparison.csv')
    FEATURE_IMPORTANCE = os.path.join(DATA_DIR, 'feature_importance.csv')
    
    # SHAP visualizations
    SHAP_SUMMARY_BASELINE = os.path.join(DATA_DIR, 'shap_summary_baseline.png')
    SHAP_IMPORTANCE_BASELINE = os.path.join(DATA_DIR, 'shap_importance_baseline.png')
    SHAP_SUMMARY_FAIR = os.path.join(DATA_DIR, 'shap_summary_fair.png')
    SHAP_IMPORTANCE_FAIR = os.path.join(DATA_DIR, 'shap_importance_fair.png')
    SHAP_INDIVIDUAL = os.path.join(DATA_DIR, 'shap_individual_example.png')
    
    # ============================================
    # MODEL FILES
    # ============================================
    BASELINE_MODEL = os.path.join(MODEL_DIR, 'baseline_model.pkl')
    FAIR_MODEL = os.path.join(MODEL_DIR, 'fair_model.pkl')
    SCALER = os.path.join(MODEL_DIR, 'scaler.pkl')
    
    # ============================================
    # DATASET INFORMATION
    # ============================================
    TARGET_COLUMN = 'default payment next month'
    SENSITIVE_ATTRIBUTE = 'SEX'  # 1=Male, 2=Female
    
    # Categorical features
    CATEGORICAL_FEATURES = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Numeric features to be scaled
    NUMERIC_FEATURES = [
        'LIMIT_BAL', 'AGE',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    # All feature names (must match training order)
    ALL_FEATURES = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    # ============================================
    # MODEL PARAMETERS
    # ============================================
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Logistic Regression parameters
    LOGISTIC_MAX_ITER = 1000
    LOGISTIC_RANDOM_STATE = 42
    
    # ============================================
    # FAIRNESS THRESHOLDS
    # ============================================
    # Disparate Impact acceptable range (80% rule)
    DISPARATE_IMPACT_MIN = 0.8
    DISPARATE_IMPACT_MAX = 1.25
    
    # Demographic Parity threshold (closer to 0 is better)
    DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
    
    # ============================================
    # WEB APPLICATION SETTINGS
    # ============================================
    FLASK_HOST = '127.0.0.1'
    FLASK_PORT = 5000
    FLASK_DEBUG = True
    
    # Default values for web form (realistic averages)
    DEFAULT_VALUES = {
        'LIMIT_BAL': 50000,
        'AGE': 35,
        'PAY_0': -1,  # Paid on time
        'PAY_2': -1,
        'PAY_3': -1,
        'PAY_4': -1,
        'PAY_5': -1,
        'PAY_6': -1,
        'BILL_AMT1': 50000,
        'BILL_AMT2': 48000,
        'BILL_AMT3': 46000,
        'BILL_AMT4': 44000,
        'BILL_AMT5': 42000,
        'BILL_AMT6': 40000,
        'PAY_AMT1': 2000,
        'PAY_AMT2': 2000,
        'PAY_AMT3': 2000,
        'PAY_AMT4': 2000,
        'PAY_AMT5': 2000,
        'PAY_AMT6': 2000
    }
    
    # ============================================
    # LOGGING CONFIGURATION
    # ============================================
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_LEVEL = 'INFO'
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.MODEL_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("✅ All necessary directories created")
    
    @classmethod
    def verify_files_exist(cls):
        """Check if essential files exist"""
        essential_files = {
            'Raw Data': cls.RAW_DATA,
            'Cleaned Data': cls.CLEAN_DATA,
            'Baseline Model': cls.BASELINE_MODEL,
            'Fair Model': cls.FAIR_MODEL,
            'Scaler': cls.SCALER
        }
        
        print("\n" + "=" * 60)
        print("FILE VERIFICATION")
        print("=" * 60)
        
        all_exist = True
        for name, path in essential_files.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"{status} {name}: {exists}")
            if not exists:
                all_exist = False
        
        print("=" * 60)
        
        return all_exist
    
    @classmethod
    def get_info(cls):
        """Display configuration information"""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Model Directory: {cls.MODEL_DIR}")
        print(f"Target Column: {cls.TARGET_COLUMN}")
        print(f"Sensitive Attribute: {cls.SENSITIVE_ATTRIBUTE}")
        print(f"Test Size: {cls.TEST_SIZE}")
        print(f"Random State: {cls.RANDOM_STATE}")
        print(f"Flask Port: {cls.FLASK_PORT}")
        print("=" * 60 + "\n")


# For quick testing
if __name__ == "__main__":
    # Display configuration
    Config.get_info()
    
    # Create directories
    Config.create_directories()
    
    # Verify files
    Config.verify_files_exist()
