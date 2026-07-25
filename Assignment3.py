import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("titanic.txt")

# Fill Missing Values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Label Encoding for Sex
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

print("After Label Encoding:")
print(df[["Sex"]].head())

# One-Hot Encoding for Embarked
df = pd.get_dummies(df, columns=["Embarked"])

print("\nAfter One-Hot Encoding:")
print(df.head())

# Save Dataset
df.to_csv("titanic_encoded.csv", index=False)