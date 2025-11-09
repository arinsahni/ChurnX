import sys
import os
import streamlit as st
import pandas as pd

# 🔧 Append the source path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipeline import churn_pipeline

st.set_page_config(page_title="ChurnX Dashboard", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: teal;'>📉 ChurnX: Customer Churn Predictor</h1>
    <hr>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Upload CSV", "Data Overview", "Model Training"])

uploaded_file = st.file_uploader("📂 Upload your Telco CSV", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("❌ The uploaded CSV file is empty.")
            st.stop()

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # === Section: Upload CSV ===
    if section == "Upload CSV":
        st.subheader("📎 Raw Uploaded Data")
        st.dataframe(df.head())

    # === Section: Data Overview ===
    elif section == "Data Overview":
        st.subheader("📊 Data Overview")
        st.write("Shape:", df.shape)
        if "Churn" in df.columns:
            st.write("Churn Distribution:")
            st.bar_chart(df["Churn"].value_counts())
        else:
            st.warning("⚠️ 'Churn' column not found in dataset.")

    # === Section: Model Training ===
    elif section == "Model Training":
        st.subheader("🧠 Train & Evaluate Churn Model")
        with st.spinner("⚙️ Running full churn pipeline..."):
            try:
                model, X_test, y_test, fig, df_preds = churn_pipeline(df, return_preds=True)
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                st.stop()

        st.success("✅ Model trained and predictions generated!")

        st.subheader("📉 Confusion Matrix")
        st.pyplot(fig)

        st.subheader("📈 Predictions")
        st.dataframe(df_preds)

        csv = df_preds.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv,
            file_name="predicted_churn.csv",
            mime="text/csv"
        )

else:
    st.info("📝 Please upload a dataset to begin.")