# src/pipeline.py

import pandas as pd
from preprocess import encode_features
from model import train_and_evaluate, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def churn_pipeline(data, model_type="rf", return_preds=False):
    """
    Full churn prediction pipeline.

    Params:
    - data: can be a pandas DataFrame, file-like object, or CSV path string
    - model_type: 'rf' or 'lr'
    - return_preds: If True, returns predictions DataFrame

    Returns:
    - model, X_test, y_test, fig (confusion matrix)
    - Optionally: df with predictions
    """
    print("🔄 Loading and preprocessing dataset...")

    # ✅ Handle various input types
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif hasattr(data, 'read'):  # Streamlit file-like object
        data.seek(0)
        df = pd.read_csv(data)
    else:
        df = pd.read_csv(data)

    print("📋 Columns in file:", df.columns.tolist())

    # 🔍 Normalize churn column name
    possible_targets = ["Churn", "churn", "Customer_Churn", "Exited"]
    target_col = next((col for col in df.columns if col.strip() in possible_targets), None)

    if not target_col:
        raise ValueError("❌ 'Churn' column not found. Expected one of: " + ", ".join(possible_targets))

    # ✅ Rename to 'Churn' for consistency
    if target_col != "Churn":
        df["Churn"] = df[target_col]

    df_encoded = encode_features(df)

    print("🚀 Training model...")
    model, X_test, y_test = train_and_evaluate(df_encoded, model_type=model_type)

    print("🧠 Plotting confusion matrix...")
    fig = plot_confusion_matrix(model, X_test, y_test)

    if return_preds:
        print("📈 Generating predictions...")
        predictions = model.predict(df_encoded.drop("Churn", axis=1))
        df_out = df.copy()
        df_out["Predicted Churn"] = predictions
        return model, X_test, y_test, fig, df_out

    return model, X_test, y_test, fig


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import os

    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # ✅ Works in standalone mode too
    model, X_test, y_test, fig, df_out = churn_pipeline(csv_path, return_preds=True)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/confusion_matrix.png")
    df_out.to_csv("outputs/predicted_churn.csv", index=False)
    print("✅ Outputs saved to /outputs/")
