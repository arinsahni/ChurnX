# src/explain.py

import os
import matplotlib.pyplot as plt

# safer relative imports: works with `python -m src.explain`
try:
    from preprocess import load_telco_data, encode_features
    from model import train_and_evaluate
except ImportError:
    from preprocess import load_telco_data, encode_features
    from model import train_and_evaluate


def explain_model_shap(model, X_train, X_test, plot_path="outputs"):
    """
    Generates SHAP values and plots for global and local explanations.
    Saves plots as PNGs inside `plot_path`.

    Returns:
        shap_values (shap.Explanation) or None if SHAP failed
    """
    print("🔥 SHAP function started")

    try:
        import shap
    except ImportError:
        print("❌ SHAP not installed. Run: pip install shap matplotlib")
        return None

    os.makedirs(plot_path, exist_ok=True)

    try:
        print("🔄 Creating SHAP explainer...")
        
        # 🟢 FIX: Use shap.TreeExplainer instead of shap.Explainer
        # This allows the 'check_additivity' argument to be correctly passed.
        explainer = shap.TreeExplainer(model = model, data= X_train,) 

        print("🔍 Computing SHAP values...")
        shap_values = explainer.shap_values(X_test)
        
        # ... (Rest of your plotting code)

        # --- Global Feature Importance ---
        print("📊 Saving global SHAP bar plot...")
        plt.figure()
        shap.plots.bar(shap_values, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_path, "shap_global_importance.png"), bbox_inches="tight"
        )
        plt.close()
        print("✅ Global importance plot saved")

        # --- Local Explanation (first sample) ---
        print("📉 Saving local waterfall plot...")
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_path, "shap_waterfall_first_sample.png"),
            bbox_inches="tight",
        )
        plt.close()
        print("✅ Local waterfall plot saved")

        print("🏁 SHAP phase completed")
        return shap_values

    except Exception as e:
        print(f"❌ SHAP error: {e}")
        return None


if __name__ == "__main__":
    print("📦 Loading dataset...")
    csv_path = "/Users/eva/Desktop/Churn data/data/WA_Fn-UseC_-Telco-Customer-Churn copy.csv"
    df = load_telco_data(csv_path)
    df_encoded = encode_features(df)

    print("🧠 Training model...")
    model, X_test, y_test = train_and_evaluate(df_encoded, model_type="rf")
    X_train = df_encoded.drop("Churn", axis=1)

    print("🔍 Explaining model...")
    shap_vals = explain_model_shap(
        model, X_train=X_train, X_test=X_test, plot_path="outputs"
    )

    if shap_vals is None:
        print("⚠️ Skipped SHAP plots due to an error (see logs above).")
    else:
        print("✅ SHAP values computed and plots saved to ./outputs")
