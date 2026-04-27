import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("=== SMART FERTILIZER MODEL TRAINING ===")

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_csv("fertilizer.csv")

print("\nDataset Shape:", df.shape)
print(df.head())

# --------------------------
# HANDLE MISSING VALUES
# --------------------------
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# --------------------------
# ENCODE CATEGORICAL DATA
# --------------------------
encoders = {}

for col in df.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nEncoded Columns:", list(encoders.keys()))

# --------------------------
# FEATURE & TARGET
# --------------------------
if 'Fertilizer Name' not in df.columns:
    raise Exception("Target column 'Fertilizer Name' not found")

X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

print("\nFeatures:", X.columns.tolist())

# --------------------------
# SPLIT DATA
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# MODEL TRAINING
# --------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)

# --------------------------
# EVALUATION
# --------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("\n=== RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Cross-validation mean: {cv_scores.mean():.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# SAVE MODEL
# --------------------------
joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("\nModel and encoders saved!")

# --------------------------
# SAMPLE PREDICTIONS
# --------------------------
print("\n=== SAMPLE PREDICTIONS ===")
sample = X_test.head(3)
preds = model.predict(sample)

for i, p in enumerate(preds):
    fert = encoders['Fertilizer Name'].inverse_transform([p])[0]
    print(f"Sample {i+1}: Recommended Fertilizer = {fert}")

# --------------------------
# VISUALIZATION
# --------------------------

# Confusion Matrix
plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# Accuracy Plot (CV Scores)
plt.figure(figsize=(7,5))
sns.lineplot(x=range(1,6), y=cv_scores, marker="o")
plt.title("Cross Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.grid()
plt.show()

print("\nTraining completed successfully!")