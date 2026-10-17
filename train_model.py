
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
DATA_PATH = "Crop_recommendation.csv"
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Clean dataset
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"] != ""]

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# 3. Separate features and target
target_column = "label"
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert feature columns to numeric and handle invalid values
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))

# Remove rows where all feature values are missing
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

if X.empty:
    raise ValueError("No usable feature data found in the CSV file.")

if y.nunique() < 2:
    raise ValueError("At least two crop categories are required.")

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of crop categories:", y.nunique())

# 5. Create preprocessing and model pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# 6. Train the model
pipeline.fit(X_train, y_train)

# 7. Evaluate on unseen test data
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Display confusion matrix
labels = pipeline.named_steps["classifier"].classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot(xticks_rotation=90, cmap="Blues")
plt.title("Crop Recommendation - Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Display feature importance
importance = pd.Series(
    pipeline.named_steps["classifier"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.sort_values().plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 10. Save model and metadata
os.makedirs("model", exist_ok=True)

joblib.dump(pipeline, "model/crop_model.pkl")
joblib.dump(list(X.columns), "model/features.pkl")

print("\nModel saved successfully!")
print("Model path: model/crop_model.pkl")
print("Features path: model/features.pkl")
