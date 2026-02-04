import pandas as pd

df = pd.read_csv(
    r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\default of credit card clients.csv',
    header=1
)
# Check for missing values
print(df.isnull().sum())
print(df.shape)
# Check duplicate rows
print("Duplicates before removal:", df.duplicated().sum())
# Remove duplicates if any
df = df.drop_duplicates()
print("Duplicates after removal:", df.duplicated().sum())
# Drop ID column
df = df.drop(columns=['ID'])
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
from sklearn.preprocessing import StandardScaler

numeric_features = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
]

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
# Define target variable
target = 'default payment next month'

# Define sensitive feature
sensitive_feature = 'SEX'

X = df.drop(columns=[target])
y = df[target]
# Save cleaned dataset
df.to_csv(
    r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\cleaned_credit_data.csv',
    index=False
)

