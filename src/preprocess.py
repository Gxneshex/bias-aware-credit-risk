import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    """
    Load and preprocess the credit card default dataset
    """
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(
        r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\default of credit card clients.csv',
        header=1
    )
    
    print(f"\n📊 Original dataset shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    # Check duplicates
    duplicates_before = df.duplicated().sum()
    print(f"Duplicates before removal: {duplicates_before}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_after = df.duplicated().sum()
    print(f"Duplicates after removal: {duplicates_after}")
    
    # Drop ID column
    df = df.drop(columns=['ID'])
    
    # Define features
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numeric_features = [
        'LIMIT_BAL', 'AGE',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    # Normalize numeric features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    print(f"\n✅ Final dataset shape: {df.shape}")
    print(f"✅ Features normalized: {len(numeric_features)}")
    
    # Save cleaned dataset
    output_path = r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\cleaned_credit_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Cleaned data saved to: cleaned_credit_data.csv")
    
    return df

if __name__ == "__main__":
    preprocess_data()
