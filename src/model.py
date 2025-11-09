# src/model.py

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def plot_confusion_matrix(model, X_test, y_test):
    """
    Returns a matplotlib figure of the confusion matrix for display in Streamlit.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig

def train_and_evaluate(df, model_type="rf"):
    """
    Trains model on preprocessed dataframe and returns trained model, test features, and labels.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = (
        RandomForestClassifier(n_estimators=100, random_state=42)
        if model_type == "rf"
        else LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))

    return model, X_test, y_test