# ==========================================
# 1. IMPORT LIBRARIES (Always at the top)
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 2. LOAD & CLEAN DATA
# ==========================================
# Make sure your CSV file is in the same folder as this .py file
df = pd.read_csv("Breast_Cancer.csv")

# Prepare target (y) and features (X)
y = df["Status"]
X = df.drop("Status", axis=1)

# Convert text to numbers
X_encoded = pd.get_dummies(X)
y_encoded = y.map({"Alive": 1, "Dead": 0})

# REMOVE 'Survival Months' to prevent cheating (Data Leakage)
X_no_leak = X_encoded.drop(columns=["Survival Months"])

# ==========================================
# 3. SPLIT & SCALE
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_no_leak, y_encoded, 
    test_size=0.2, 
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
# We use 'balanced' weights because the 'Dead' group is small
model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# ==========================================
# 5. TEST & SHOW RESULTS
# ==========================================
predictions = model.predict(X_test_scaled)

print("--- MODEL PERFORMANCE REPORT ---")
print(f"Accuracy Score: {accuracy_score(y_test, predictions):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# Show top 5 most important features
feature_importance = pd.Series(
    model.coef_[0], 
    index=X_no_leak.columns
).sort_values(key=abs, ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))
