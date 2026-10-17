
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Load and prepare dataset
df = pd.read_csv("Crop_recommendation.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=["label"])

X = df.drop(columns=["label"]).apply(pd.to_numeric, errors="coerce")
y = df["label"].astype(str).str.strip()

X = X.replace([float("inf"), float("-inf")], float("nan"))
valid = X.notna().any(axis=1) & y.ne("")
X, y = X.loc[valid], y.loc[valid]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    ),
    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42)
    ),
    "KNN": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC()
    )
}

# Train, evaluate and compare
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    results.append({
        "Model": name,
        "Test Accuracy (%)": round(accuracy * 100, 2),
        "Macro F1 (%)": round(macro_f1 * 100, 2),
        "5-Fold CV Mean (%)": round(cv_scores.mean() * 100, 2),
        "CV Std Dev (%)": round(cv_scores.std() * 100, 2)
    })

results_df = pd.DataFrame(results).sort_values(
    "Macro F1 (%)", ascending=False
)

print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)
print("\nComparison saved to model_comparison.csv")
