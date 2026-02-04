import pandas as pd

df = pd.read_csv(
    r'C:\Users\rsury\OneDrive\Desktop\bias-aware-credit-risk\data\default of credit card clients.csv',
    header=1
)

print(df.head())
print(df.columns)
print(df.info())
