import sys
import os
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Add 'src/' folder to import path
sys.path.append(os.path.abspath("src"))

from preprocess import load_telco_data, encode_features
from model import train_and_evaluate

# --- Step 1: Load and encode data ---
print("📦 Loading dataset...")
df = load_telco_data("data/WA_Fn-UseC_-Telco-Customer-Churn copy.csv")

# Optional: sample for memory efficiency (can remove later)
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

df_encoded = encode_features(df)
X = df_encoded.drop("Churn", axis=1)

# --- Step 2: Train the model ---
print("🧠 Training model...")
model, X_test, y_test = train_and_evaluate(df_encoded, model_type="rf")
print("✅ Model training complete.")

# --- Step 3: Permutation Importance ---
print("📊 Computing permutation importance on test subset...")
X_test_small = X_test.iloc[:100]
y_test_small = y_test.iloc[:100]

result = permutation_importance(
    model, X_test_small, y_test_small, n_repeats=3, random_state=42
)
sorted_idx = result.importances_mean.argsort()[::-1]

# --- Step 4: Save plot ---
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.barh(
    X_test.columns[sorted_idx][:15][::-1],
    result.importances_mean[sorted_idx][:15][::-1]
)
plt.xlabel("Permutation Importance")
plt.title("Top 15 Features")
plt.tight_layout()
plt.savefig("outputs/permutation_importance.png")
plt.close()

print("✅ Permutation importance plot saved to outputs/permutation_importance.png")
