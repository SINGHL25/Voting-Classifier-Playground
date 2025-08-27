
# app.py
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About Voting Classifier")
st.sidebar.markdown("""
**Voting Classifier** combines multiple models:
- **Logistic Regression (LR)** – good for linear relationships  
- **Support Vector Machine (SVM)** – good for non-linear boundaries  
- **Random Forest (RF)** – tree-based ensemble  

**Hard voting:** each model casts a vote for a class; majority wins.  
Helps improve robustness and reduce overfitting.
""")

# -------------------------------
# Dataset
# -------------------------------
st.title("🗳️ Voting Classifier Playground (2D & 3D)")
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
use_example = st.checkbox("Use Example Dataset (iris.csv)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_example:
    df = pd.read_csv("example_data.csv")
else:
    st.info("Upload CSV or use example dataset.")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Target selection
target_col = st.selectbox("Select target column", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# Train/Test split
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.number_input("Random Seed", 0, 9999, 42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
)

# Optional scaling
scale_features = st.checkbox("Standardize features (recommended for decision boundaries)", True)
if scale_features:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
else:
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

# -------------------------------
# GridSearchCV for RandomForest
# -------------------------------
st.subheader("⚙️ Random Forest Hyperparameter Tuning")
n_estimators_options = st.multiselect("n_estimators", [50, 100, 200], default=[100])
max_depth_options = st.multiselect("max_depth", [None, 5, 10, 20], default=[None, 10])

if st.button("Run GridSearchCV for Random Forest"):
    rf = RandomForestClassifier(random_state=random_state)
    param_grid = {"n_estimators": n_estimators_options, "max_depth": max_depth_options}
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    st.success(f"Best Parameters: {grid.best_params_}")
    best_rf = grid.best_estimator_
else:
    best_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state)

# -------------------------------
# Voting Classifier
# -------------------------------
lr = LogisticRegression(max_iter=1000, random_state=random_state)
svm = SVC(probability=False, random_state=random_state)
voting_clf = VotingClassifier(estimators=[("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard")

if st.button("Train Voting Classifier"):
    voting_clf.fit(X_train_scaled, y_train)
    y_pred = voting_clf.predict(X_test_scaled)
    st.subheader("📊 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(voting_clf, "voting_classifier.pkl")
    st.success("✅ Model saved as `voting_classifier.pkl`")

# -------------------------------
# Load Saved Model
# -------------------------------
if st.button("Load Existing Model"):
    try:
        loaded_model = joblib.load("voting_classifier.pkl")
        st.success("Loaded saved model successfully!")
        y_pred = loaded_model.predict(X_test_scaled)
        st.write("**Accuracy (Loaded Model):**", accuracy_score(y_test, y_pred))
    except FileNotFoundError:
        st.error("No saved model found. Train one first!")

# -------------------------------
# 2D Decision Boundary
# -------------------------------
if X.shape[1] >= 2:
    st.subheader("📐 2D Decision Boundary")
    all_features = list(X.columns)
    f1 = st.selectbox("Feature 1 (2D)", all_features, index=0)
    f2 = st.selectbox("Feature 2 (2D)", all_features, index=1 if len(all_features) > 1 else 0)

    X2_train = X_train_scaled[[f1, f2]].values
    X2_test = X_test_scaled[[f1, f2]].values
    clf2 = RandomForestClassifier(n_estimators=best_rf.n_estimators,
                                  max_depth=best_rf.max_depth,
                                  random_state=random_state)
    clf2.fit(X2_train, y_train)

    # Meshgrid
    xx, yy = np.meshgrid(np.linspace(X2_train[:,0].min()-1, X2_train[:,0].max()+1, 200),
                         np.linspace(X2_train[:,1].min()-1, X2_train[:,1].max()+1, 200))
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.contourf(xx, yy, Z, alpha=0.25)
    ax.scatter(X2_train[:,0], X2_train[:,1], c=y_train, edgecolor="k")
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title("2D Decision Boundary")
    st.pyplot(fig)

# -------------------------------
# 3D Decision Boundary (Plotly)
# -------------------------------
if X.shape[1] >= 3:
    st.subheader("🖥️ 3D Decision Boundary")
    f1_3d = st.selectbox("Feature 1 (3D)", all_features, index=0)
    f2_3d = st.selectbox("Feature 2 (3D)", all_features, index=1)
    f3_3d = st.selectbox("Feature 3 (3D)", all_features, index=2)

    X3_train = X_train_scaled[[f1_3d, f2_3d, f3_3d]].values
    X3_test = X_test_scaled[[f1_3d, f2_3d, f3_3d]].values
    rf3 = RandomForestClassifier(n_estimators=best_rf.n_estimators,
                                 max_depth=best_rf.max_depth,
                                 random_state=random_state)
    rf3.fit(X3_train, y_train)

    # Coarse mesh
    x_range = np.linspace(X3_train[:,0].min()-1, X3_train[:,0].max()+1, 20)
    y_range = np.linspace(X3_train[:,1].min()-1, X3_train[:,1].max()+1, 20)
    z_range = np.linspace(X3_train[:,2].min()-1, X3_train[:,2].max()+1, 20)
    mesh = np.array(list(itertools.product(x_range, y_range, z_range)))
    Z3 = rf3.predict(mesh)

    fig = go.Figure()
    for cls in np.unique(y_train):
        mask = y_train == cls
        fig.add_trace(go.Scatter3d(x=X3_train[mask,0], y=X3_train[mask,1], z=X3_train[mask,2],
                                   mode='markers', marker=dict(size=4, opacity=0.8),
                                   name=f'Train Class {cls}'))
    for cls in np.unique(y_test):
        mask = y_test == cls
        fig.add_trace(go.Scatter3d(x=X3_test[mask,0], y=X3_test[mask,1], z=X3_test[mask,2],
                                   mode='markers', marker=dict(symbol='x', size=5, opacity=0.8),
                                   name=f'Test Class {cls}'))
    fig.update_layout(scene=dict(xaxis_title=f1_3d, yaxis_title=f2_3d, zaxis_title=f3_3d),
                      title="3D Decision Boundary (RF in Voting Classifier)")
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("💡 Tip: Toggle standardization for smoother boundaries. Save/load models to replay without retraining.")


