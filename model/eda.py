
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder for saving graphs
os.makedirs("model/eda_results", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_disease_dataset.csv")

# Separate symptoms and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print("=" * 55)
print("AI DISEASE PREDICTION SYSTEM - DATA ANALYSIS")
print("=" * 55)

# 1. Dataset information
print("\n1. DATASET INFORMATION")

print("Dataset Shape:", df.shape)
print("Total Records:", len(df))
print("Total Symptoms:", X.shape[1])
print("Total Diseases:", y.nunique())

print("\nDataset Information:")
df.info()

# 2. Data quality checks
print("\n2. DATA QUALITY CHECKS")

print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Duplicate Symptom Patterns:", X.duplicated().sum())

print("\nUnique Feature Values:")
print("Unique Feature Values:", sorted(set(X.values.flatten())))

# 3. Disease distribution
print("\n3. DISEASE DISTRIBUTION")

disease_counts = y.value_counts()

print(disease_counts.to_string())

plt.figure(figsize=(15, 8))

disease_counts.plot(kind="bar")

plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Records")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    "model/eda_results/disease_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# 4. Top 20 common symptoms
print("\n4. TOP 20 COMMON SYMPTOMS")

symptom_counts = X.sum().sort_values(ascending=False)

print(symptom_counts.head(20).to_string())

plt.figure(figsize=(12, 7))

sns.barplot(
    x=symptom_counts.head(20).values,
    y=symptom_counts.head(20).index
)

plt.title("Top 20 Common Symptoms")
plt.xlabel("Number of Records")
plt.ylabel("Symptoms")
plt.tight_layout()

plt.savefig(
    "model/eda_results/top_symptoms.png",
    dpi=300
)

plt.show()
plt.close()

# 5. Disease sample distribution
print("\n5. DISEASE SAMPLE DISTRIBUTION")

print("Minimum Samples:", disease_counts.min())
print("Maximum Samples:", disease_counts.max())
print("Average Samples:", disease_counts.mean())

plt.figure(figsize=(9, 5))

sns.histplot(
    disease_counts.values,
    bins=10,
    kde=False
)

plt.title("Distribution of Samples Across Diseases")
plt.xlabel("Number of Records per Disease")
plt.ylabel("Number of Diseases")
plt.tight_layout()

plt.savefig(
    "model/eda_results/disease_sample_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# 6. Binary feature distribution
print("\n6. BINARY FEATURE DISTRIBUTION")

binary_counts = X.stack().value_counts().sort_index()

print(binary_counts.to_string())

plt.figure(figsize=(7, 5))

sns.barplot(
    x=binary_counts.index.astype(str),
    y=binary_counts.values
)

plt.title("Symptom Feature Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    "model/eda_results/binary_feature_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# 7. Correlation between top 15 common symptoms
print("\n7. SYMPTOM CORRELATION ANALYSIS")

top_15_symptoms = symptom_counts.head(15).index

correlation_matrix = X[top_15_symptoms].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Top 15 Symptoms")
plt.tight_layout()

plt.savefig(
    "model/eda_results/symptom_correlation.png",
    dpi=300
)

plt.show()
plt.close()

# 8. Save EDA summary
print("\n8. SAVING EDA SUMMARY")

summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Symptoms",
        "Total Diseases",
        "Missing Values",
        "Duplicate Rows",
        "Duplicate Symptom Patterns",
        "Minimum Samples per Disease",
        "Maximum Samples per Disease"
    ],
    "Value": [
        len(df),
        X.shape[1],
        y.nunique(),
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        X.duplicated().sum(),
        disease_counts.min(),
        disease_counts.max()
    ]
})

summary.to_csv(
    "model/eda_results/eda_summary.csv",
    index=False
)

# 9. Completion message
print("\n" + "=" * 55)
print("EDA COMPLETED SUCCESSFULLY!")
print("Graphs and summary saved in model/eda_results/")
print("=" * 55)
