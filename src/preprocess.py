import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_telco_data(WA_Fn_UseC_Telco_Customer_Churn_copy_csv):
    df = pd.read_csv(WA_Fn_UseC_Telco_Customer_Churn_copy_csv)
    df.dropna(inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Encode target
    # Only map if values are still "Yes"/"No"
    if df["Churn"].dtype == object or df["Churn"].isin(["Yes", "No"]).any():
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)
 
    return df


def encode_features(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded


print("Module 'preprocess' loaded successfully.")