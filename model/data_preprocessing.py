
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/symbipredict_2022.csv")

print("Original Dataset Shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Remove extra spaces from disease names
df["prognosis"] = df["prognosis"].str.strip()

# Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# Check missing values
print("\nMissing Values:", df.isnull().sum().sum())

# Check duplicate records
print("Duplicate Records:", df.duplicated().sum())

# Display cleaned dataset information
print("\nCleaned Dataset Shape:", df.shape)

print("\nTotal Disease Categories:", df["prognosis"].nunique())

print("\nRecords per Disease:")
print(df["prognosis"].value_counts())

# Separate input features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print("\nInput Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Save cleaned dataset
df.to_csv("dataset/cleaned_disease_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")
