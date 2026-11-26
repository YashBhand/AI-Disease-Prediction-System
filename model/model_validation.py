
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_disease_dataset.csv")

# 2. Separate input and output
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

print("=" * 60)
print("AI DISEASE PREDICTION SYSTEM - MODEL VALIDATION")
print("=" * 60)

print("\nDataset Shape:", df.shape)
print("Total Diseases:", y.nunique())
print("Total Records:", len(df))

# 3. Create repeated stratified cross-validation
kfold = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

print("\nValidation Strategy: Repeated Stratified 5-Fold")
print("Number of Repeats: 10")
print("Total Evaluation Folds: 50")

# 4. Define machine learning models
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

# 5. Define evaluation metrics
scoring = {
    "accuracy": "accuracy",

    "precision": make_scorer(
        precision_score,
        average="macro",
        zero_division=0
    ),

    "recall": make_scorer(
        recall_score,
        average="macro",
        zero_division=0
    ),

    "f1": make_scorer(
        f1_score,
        average="macro",
        zero_division=0
    )
}

results = []

# 6. Validate each model
for name, model in models.items():

    print("\n" + "-" * 50)
    print("Validating:", name)
    print("-" * 50)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Calculate mean and standard deviation
    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()

    precision_mean = scores["test_precision"].mean()
    recall_mean = scores["test_recall"].mean()
    f1_mean = scores["test_f1"].mean()

    train_accuracy = scores["train_accuracy"].mean()

    print("Training Accuracy:", round(train_accuracy, 4))

    print(
        "Testing Accuracy:",
        round(accuracy_mean, 4),
        "+/-",
        round(accuracy_std, 4)
    )

    print("Precision:", round(precision_mean, 4))
    print("Recall:", round(recall_mean, 4))
    print("F1 Score:", round(f1_mean, 4))

    # Store results
    results.append({
        "Algorithm": name,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy Mean": accuracy_mean,
        "Testing Accuracy Std": accuracy_std,
        "Precision": precision_mean,
        "Recall": recall_mean,
        "F1 Score": f1_mean
    })

# 7. Display model comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("REPEATED CROSS-VALIDATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

# 8. Save validation results
results_df.to_csv(
    "model/validation_results.csv",
    index=False
)

print("\nValidation results saved successfully!")
print("File: model/validation_results.csv")

print("\n" + "=" * 60)
print("MODEL VALIDATION COMPLETED!")
print("=" * 60)
