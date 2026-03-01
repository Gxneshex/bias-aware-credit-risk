import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def preprocess_data():
    """
    Load and preprocess the credit card default dataset
    """
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Load data using Config
    print(f"\n📂 Loading data from: {Config.RAW_DATA}")
    df = pd.read_csv(Config.RAW_DATA, header=1)
    
    print(f"📊 Original dataset shape: {df.shape}")
    
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
    
    # Use feature lists from Config
    print(f"\n🔧 Normalizing {len(Config.NUMERIC_FEATURES)} numeric features...")
    
    # Normalize numeric features
    scaler = StandardScaler()
    df[Config.NUMERIC_FEATURES] = scaler.fit_transform(df[Config.NUMERIC_FEATURES])
    
    print(f"✅ Final dataset shape: {df.shape}")
    print(f"✅ Features normalized: {len(Config.NUMERIC_FEATURES)}")
    
    # ⭐ CRITICAL: Save the scaler for web app use!
    print(f"\n💾 Saving scaler to: {Config.SCALER}")
    with open(Config.SCALER, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved successfully")
    
    # Save cleaned dataset
    print(f"💾 Saving cleaned data to: {Config.CLEAN_DATA}")
    df.to_csv(Config.CLEAN_DATA, index=False)
    
    print(f"\n✅ Preprocessing complete!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    # Create necessary directories first
    Config.create_directories()
    
    # Run preprocessing
    preprocess_data()
    
    # Verify files
    print("\n")
    Config.verify_files_exist()
