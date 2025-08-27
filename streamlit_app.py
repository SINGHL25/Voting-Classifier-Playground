
# app.py
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                              BaggingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.metrics import accuracy_score, classification_report

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("Algorithm Playground Info")
st.sidebar.markdown("""
This app allows you to test and visualize multiple classifiers:
- **Voting Classifier (Hard Voting)** – combines LR, SVM, RF
- **Bagging Classifier**
- **AdaBoost**
- **Gradient Boosting**
- **XGBoost (optional)**  

💡 Features:
- Train/Test split with standardization  
- GridSearchCV for automatic tuning  
- Export/Import trained models  
- 2D & 3D decision boundary visualization  
- Accuracy & Classification Report
""")

# -------------------------------
# Dataset
# -------------------------------
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

target_col = st.selectbox("Select Target Column", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# Remove rare classes
rare_classes = y.value_counts()[y.value_counts() < 2].index
if len(rare_classes) > 0:
    st.warning(f"⚠️ Classes with <2 samples will be removed: {list(rare_classes)}")
    mask = ~y.isin(rare_classes)
    X = X[mask]
    y = y[mask]

# -------------------------------
# Train/Test Split
# -------------------------------
test_size = st.slider("Test Size (%)", 10, 50, 20)/100
random_state = st.number_input("Random Seed", 0, 9999, 42)
scale_features = st.checkbox("Standardize Features", True)
if scale_features:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
else:
    X_scaled = X.copy()
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
)

# -------------------------------
# Classifier Selection
# -------------------------------
st.subheader("Choose Classifier")
classifier = st.selectbox("Classifier", [
    "Voting", "Bagging", "AdaBoost", "GradientBoosting"] + (["XGBoost"] if xgb_available else []))

# -------------------------------
# Hyperparameter Tuning Example for RF
# -------------------------------
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]}
if classifier in ["Voting", "RandomForest"]:
    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    if st.button("Run GridSearchCV (RF)"):
        grid.fit(X_train, y_train)
        st.success(f"Best RF Params: {grid.best_params_}")
        best_rf = grid.best_estimator_
    else:
        best_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state)

# -------------------------------
# Train Classifier
# -------------------------------
def train_model(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return clf, acc, report, y_pred

if st.button("Train Classifier"):
    if classifier == "Voting":
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        svm = SVC(probability=False, random_state=random_state)
        voting_clf = VotingClassifier(estimators=[("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard")
        clf, acc, report, y_pred = train_model(voting_clf)
    elif classifier == "Bagging":
        clf, acc, report, y_pred = train_model(BaggingClassifier(base_estimator=best_rf, n_estimators=10, random_state=random_state))
    elif classifier == "AdaBoost":
        clf, acc, report, y_pred = train_model(AdaBoostClassifier(base_estimator=best_rf, n_estimators=50, random_state=random_state))
    elif classifier == "GradientBoosting":
        clf, acc, report, y_pred = train_model(GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=random_state))
    elif classifier == "XGBoost" and xgb_available:
        clf, acc, report, y_pred = train_model(XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric="mlogloss", random_state=random_state))
    else:
        st.error("Classifier not implemented")
        st.stop()

    st.write(f"**Accuracy:** {acc}")
    st.text("Classification Report:")
    st.text(report)
    joblib.dump(clf, f"{classifier}_model.pkl")
    st.success(f"✅ Model trained and saved as `{classifier}_model.pkl`")

# -------------------------------
# Load Saved Model
# -------------------------------
if st.button("Load Saved Model"):
    try:
        loaded_model = joblib.load(f"{classifier}_model.pkl")
        st.success("Loaded saved model successfully!")
        y_pred = loaded_model.predict(X_test)
        st.write(f"**Accuracy (Loaded Model): {accuracy_score(y_test, y_pred)}")
    except FileNotFoundError:
        st.error("No saved model found. Train one first!")

# -------------------------------
# 2D/3D Decision Boundary
# -------------------------------
st.subheader("Decision Boundary Visualization")
if st.checkbox("Show 2D Decision Boundary"):
    if X.shape[1] >= 2:
        feat1, feat2 = st.selectbox("Feature 1", X.columns), st.selectbox("Feature 2", X.columns)
        if feat1 != feat2:
            clf2d = clf
            clf2d.fit(X_train[[feat1, feat2]], y_train)
            x_min, x_max = X_scaled[feat1].min()-1, X_scaled[feat1].max()+1
            y_min, y_max = X_scaled[feat2].min()-1, X_scaled[feat2].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig, ax = plt.subplots()
            ax.contourf(xx, yy, Z, alpha=0.3)
            ax.scatter(X_scaled[feat1], X_scaled[feat2], c=y, edgecolor='k')
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            st.pyplot(fig)
    else:
        st.warning("Need at least 2 features for 2D plot.")

if st.checkbox("Show 3D Decision Boundary"):
    if X.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        f1, f2, f3 = st.selectbox("F1", X.columns), st.selectbox("F2", X.columns), st.selectbox("F3", X.columns)
        if len({f1,f2,f3})==3:
            clf3d = clf
            clf3d.fit(X_train[[f1,f2,f3]], y_train)
            x_min, x_max = X_scaled[f1].min()-1, X_scaled[f1].max()+1
            y_min, y_max = X_scaled[f2].min()-1, X_scaled[f2].max()+1
            z_min, z_max = X_scaled[f3].min()-1, X_scaled[f3].max()+1
            xx, yy, zz = np.meshgrid(np.linspace(x_min,x_max,20),
                                     np.linspace(y_min,y_max,20),
                                     np.linspace(z_min,z_max,20))
            grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            Z = clf3d.predict(grid).reshape(xx.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_scaled[f1], X_scaled[f2], X_scaled[f3], c=y)
            ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_zlabel(f3)
            st.pyplot(fig)
    else:
        st.warning("Need at least 3 features for 3D plot.")

st.markdown("---")
st.markdown("💡 Tip: Train models, visualize boundaries, and export/import for reuse.")



