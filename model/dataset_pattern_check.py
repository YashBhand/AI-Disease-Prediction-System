
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_disease_dataset.csv")

# Separate symptoms and disease labels
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print("=" * 65)
print("DISEASE-SPECIFIC SYMPTOM PATTERN ANALYSIS")
print("=" * 65)

# 1. Count active symptoms in each record
df["total_symptoms"] = X.sum(axis=1)

print("\n1. SYMPTOMS PER RECORD")

print(df["total_symptoms"].describe())

# 2. Analyze symptom patterns disease-wise
print("\n2. DISEASE-WISE SYMPTOM ANALYSIS")

results = []

for disease in sorted(y.unique()):

    disease_data = df[df["prognosis"] == disease]

    disease_X = disease_data.drop(
        columns=["prognosis", "total_symptoms"]
    )

    results.append({
        "Disease": disease,
        "Records": len(disease_data),
        "Average Symptoms": disease_X.sum(axis=1).mean(),
        "Minimum Symptoms": disease_X.sum(axis=1).min(),
        "Maximum Symptoms": disease_X.sum(axis=1).max(),
        "Unique Patterns": len(disease_X.drop_duplicates())
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# 3. Find symptoms that occur in every record of a disease
print("\n3. COMMON SYMPTOMS WITHIN EACH DISEASE")

for disease in sorted(y.unique()):

    disease_data = X[y == disease]

    common_symptoms = disease_data.columns[
        (disease_data == 1).all(axis=0)
    ].tolist()

    print("\nDisease:", disease)
    print("Symptoms present in every record:", common_symptoms)

# 4. Save analysis
results_df.to_csv(
    "model/disease_pattern_analysis.csv",
    index=False
)

print("\n" + "=" * 65)
print("PATTERN ANALYSIS COMPLETED!")
print("Results saved to model/disease_pattern_analysis.csv")
print("=" * 65)
