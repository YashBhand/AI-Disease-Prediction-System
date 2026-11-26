
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("dataset/cleaned_disease_dataset.csv")

# Separate symptoms and disease labels
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Create only one feature:
# Total number of symptoms in each record
X_count = pd.DataFrame({
    "total_symptoms": X.sum(axis=1)
})

print("=" * 60)
print("SYMPTOM COUNT-ONLY CLASSIFICATION TEST")
print("=" * 60)

print("\nOriginal Features:", X.shape[1])
print("Features Used in This Test:", X_count.shape[1])

# Repeated stratified cross-validation
kfold = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Naive Bayes": GaussianNB()
}

results = []

# Evaluate each model using only symptom count
for name, model in models.items():

    print("\nTesting:", name)

    scores = cross_val_score(
        model,
        X_count,
        y,
        cv=kfold,
        scoring="accuracy",
        n_jobs=-1
    )

    accuracy = scores.mean()
    std = scores.std()

    print("Accuracy:", round(accuracy, 4))
    print("Standard Deviation:", round(std, 4))

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Standard Deviation": std
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\n===== SYMPTOM COUNT-ONLY RESULTS =====")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "model/symptom_count_test_results.csv",
    index=False
)

print("\nResults saved successfully!")
print("=" * 60)
