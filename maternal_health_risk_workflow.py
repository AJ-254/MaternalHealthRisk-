import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# === Step 1: Load Dataset ===
data_path = "Maternal Health Risk Data Set.csv"
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print(df.head())

# === Step 2: Handle Missing Values ===
df = df.dropna()

# === Step 3: Define Features and Target ===
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

# Convert target labels to numeric (Low=0, Mid=1, High=2)
y = y.map({"low risk": 0, "mid risk": 1, "high risk": 2})

# === Step 4: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 5: Handle Class Imbalance with SMOTE ===
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_resampled.value_counts().to_dict())

# === Step 6: Normalize the Features ===
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# === Step 7: Train Model (Random Forest) ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_resampled, y_resampled)

# === Step 8: Evaluate ===
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Step 9: Save Model and Scaler ===
joblib.dump(model, "maternal_health_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully ✅")