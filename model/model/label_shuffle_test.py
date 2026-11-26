import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_disease_dataset.csv")

X = df.drop("prognosis", axis=1)
y = df["prognosis"].copy()

print("=" * 60)
print("LABEL SHUFFLING EXPERIMENT")
print("=" * 60)

print("\nTotal Records:", len(df))
print("Total Diseases:", y.nunique())
print("Expected Chance Accuracy:", round(1 / y.nunique(), 4))

# Shuffle disease labels
y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
X = X.reset_index(drop=True)

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Naive Bayes": GaussianNB()
}

for name, model in models.items():

    print("\nTesting:", name)

    scores = cross_val_score(
        model,
        X,
        y_shuffled,
        cv=kfold,
        scoring="accuracy",
        n_jobs=-1
    )

    print("Fold Accuracies:", scores)
    print("Mean Accuracy:", round(scores.mean(), 4))

print("\n" + "=" * 60)
print("LABEL SHUFFLING TEST COMPLETED!")
print("=" * 60)