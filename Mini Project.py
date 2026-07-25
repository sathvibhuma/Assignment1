import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("titanic.txt")

print("Original Dataset")
print(df.head())

# Dataset Info
print("\nDataset Info")
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin (too many missing values)
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# Encode Sex
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# One-Hot Encode Embarked
df = pd.get_dummies(df, columns=["Embarked"])

print("\nCleaned Dataset")
print(df.head())

print("\nRemaining Missing Values")
print(df.isnull().sum())

# Save Clean Dataset
df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nData Cleaning Completed Successfully!")