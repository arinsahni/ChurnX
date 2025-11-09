import os
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.preprocess import load_telco_data, encode_features
    from src.model import train_and_evaluate
except ImportError:
    from preprocess import load_telco_data, encode_features
    from model import train_and_evaluate

from sklearn.inspection import permutation_importance


if __name__ == "__main__":
    print("📦 Loading dataset...")
    csv_path = "/Users/arinsahni/Desktop/ml projects and resources/Churn data/data/WA_Fn-UseC_-Telco-Customer-Churn copy.csv"
    df = load_telco_data(csv_path)

    # 🧹 Sample to reduce memor
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    df_encoded = encode_features(df)

    print("🧠 Training model...")
    model, X_test, y_test = train_and_evaluate(df_encoded, model_type="rf")
    X_train = df_encoded.drop("Churn", axis=1)

    print("✅ Model training complete.")

    # 🧹 Free memory
    gc.collect()

    # 🔍 Permutation Importance
    print("📊 Computing permutation importance on small test set...")
    X_test_small = X_test.iloc[:100]
    y_test_small = y_test.iloc[:100]

    result = permutation_importance(model, X_test_small, y_test_small, n_repeats=3, random_state=42)

    sorted_idx = result.importances_mean.argsort()[::-1]
    os.makedirs("outputs", exist_ok=True)

    # ✅ Now create and save the actual plot
    plt.figure(figsize=(8, 5))
    plt.barh(X_test.columns[sorted_idx][:15][::-1], result.importances_mean[sorted_idx][:15][::-1])
    plt.xlabel("Permutation Importance")
    plt.title("Top 15 Features")
    plt.tight_layout()
    plt.savefig("outputs/permutation_importance.png")
    plt.close()

    print("Permutation importance plot saved to outputs/permutation_importance.png")
