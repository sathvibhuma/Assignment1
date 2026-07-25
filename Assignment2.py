import pandas as pd

# Load dataset
df = pd.read_csv("titanic.txt")

# Fill Age with Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Fare with Mean
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

# Fill Embarked with Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Check Missing Values
print(df.isnull().sum())

# Save Clean Dataset
df.to_csv("titanic_cleaned.csv", index=False)

print("Missing values handled successfully!")