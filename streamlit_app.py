
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About the Voting Classifier")
st.sidebar.markdown("""
The **Voting Classifier** combines predictions from multiple models:
- **Logistic Regression (LR)** – good for linear relationships  
- **Support Vector Machine (SVM)** – good for non-linear boundaries  
- **Random Forest (RF)** – strong tree-based ensemble  

With **hard voting**, each model casts a "vote" for a class, and the majority wins.  
This helps improve robustness and reduce overfitting.
""")

# -------------------------------
# Main App
# -------------------------------
st.title("🗳️ Voting Classifier App with Auto-Tuning & Export")

# File Upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
use_example = st.checkbox("Use Example Dataset (iris.csv)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_example:
    df = pd.read_csv("example_data.csv")
else:
    st.info("Upload a CSV file or use the example dataset.")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Target column selection
target_col = st.selectbox("Select target column", df.columns)

# Features and Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-Test Split
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.number_input("Random Seed", 0, 9999, 42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# -------------------------------
# GridSearchCV for RandomForest
# -------------------------------
st.subheader("⚙️ Hyperparameter Tuning (Random Forest)")

param_grid = {
    "n_estimators": st.multiselect("n_estimators", [50, 100, 200], default=[100]),
    "max_depth": st.multiselect("max_depth", [None, 5, 10, 20], default=[None, 10]),
}

if st.button("Run GridSearchCV for Random Forest"):
    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    st.success(f"Best Parameters: {grid.best_params_}")
    best_rf = grid.best_estimator_
else:
    best_rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=random_state
    )

# -------------------------------
# Voting Classifier
# -------------------------------
lr = LogisticRegression(max_iter=1000, random_state=random_state)
svm = SVC(probability=False, random_state=random_state)

voting_clf = VotingClassifier(
    estimators=[("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard"
)

if st.button("Train Voting Classifier"):
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    st.subheader("📊 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(voting_clf, "voting_classifier.pkl")
    st.success("✅ Model trained and saved as `voting_classifier.pkl`")

# -------------------------------
# Load Saved Model
# -------------------------------
if st.button("Load Existing Model"):
    try:
        loaded_model = joblib.load("voting_classifier.pkl")
        st.success("Loaded saved model successfully!")
        y_pred = loaded_model.predict(X_test)
        st.write("**Accuracy (Loaded Model):**", accuracy_score(y_test, y_pred))
    except FileNotFoundError:
        st.error("No saved model found. Train one first!")

# Footer
st.markdown("---")
st.markdown("💡 Tip: Try uploading your own dataset, or use the provided **example_data.csv** for testing.")
