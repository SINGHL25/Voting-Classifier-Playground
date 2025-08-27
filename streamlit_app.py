
# app.py
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
from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About the Voting Classifier")
st.sidebar.markdown("""
The **Voting Classifier** combines predictions from multiple models:
- **Logistic Regression (LR)** – linear model
- **Support Vector Machine (SVM)** – non-linear boundaries
- **Random Forest (RF)** – tree-based ensemble

With **hard voting**, each model casts a "vote" for a class, and the majority wins.
This improves robustness and reduces overfitting.
""")

# -------------------------------
# Main App
# -------------------------------
st.title("🗳️ Voting Classifier Playground")

# File Upload or Example
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
use_example = st.checkbox("Use Example Dataset (Iris)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_example:
    df = pd.read_csv("example_data.csv")
else:
    st.info("Upload a CSV file or use the example dataset.")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Select target column
target_col = st.selectbox("Select Target Column", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# Train/Test Split
# -------------------------------
st.subheader("⚙️ Train/Test Split & Scaling")
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.number_input("Random Seed", 0, 9999, 42)
scale_features = st.checkbox("Standardize Features", value=False)

X_model = X.copy()
if scale_features:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_model[:] = scaler.fit_transform(X_model.values)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=test_size, random_state=random_state, stratify=y
    )
except ValueError:
    st.error("Some classes have too few samples for stratified split. Reduce stratify or check dataset.")
    st.stop()

# -------------------------------
# GridSearchCV for Random Forest
# -------------------------------
st.subheader("⚙️ Random Forest Hyperparameter Tuning (Optional)")

param_grid = {
    "n_estimators": st.multiselect("n_estimators", [50, 100, 200], default=[100]),
    "max_depth": st.multiselect("max_depth", [None, 5, 10, 20], default=[None, 10])
}

if st.button("Run GridSearchCV for RF"):
    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    st.success(f"Best Parameters: {grid.best_params_}")
    best_rf = grid.best_estimator_
else:
    best_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state)

# -------------------------------
# Voting Classifier
# -------------------------------
st.subheader("🚀 Train Voting Classifier")

lr = LogisticRegression(max_iter=1000, random_state=random_state)
svm = SVC(probability=False, random_state=random_state)

voting_clf = VotingClassifier(
    estimators=[("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard"
)

if st.button("Train Model"):
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
# Load Existing Model
# -------------------------------
if st.button("Load Saved Model"):
    try:
        loaded_model = joblib.load("voting_classifier.pkl")
        st.success("Loaded saved model successfully!")
        y_pred = loaded_model.predict(X_test)
        st.write("**Accuracy (Loaded Model):**", accuracy_score(y_test, y_pred))
    except FileNotFoundError:
        st.error("No saved model found. Train one first!")

# -------------------------------
# Decision Tree Visualization
# -------------------------------
st.subheader("🌳 Random Forest Tree (first tree only)")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(best_rf.estimators_[0], filled=True, rounded=True,
          feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], ax=ax)
st.pyplot(fig)

# -------------------------------
# Feature Importances
# -------------------------------
st.subheader("📌 Feature Importances")
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(len(importances)), importances[indices])
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha="right")
ax.set_ylabel("Importance")
ax.set_title("Feature Importances")
st.pyplot(fig)

# -------------------------------
# 2D Decision Boundary
# -------------------------------
st.subheader("📈 2D Decision Boundary")
if X.shape[1] >= 2:
    f1 = st.selectbox("Feature 1", X.columns, key="2d1")
    f2 = st.selectbox("Feature 2", X.columns, index=1, key="2d2")
    if f1 != f2:
        from matplotlib.colors import ListedColormap

        clf2 = VotingClassifier([("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard")
        clf2.fit(X_train[[f1, f2]], y_train)

        x_min, x_max = X_model[f1].min() - 1, X_model[f1].max() + 1
        y_min, y_max = X_model[f2].min() - 1, X_model[f2].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter = ax.scatter(X_model[f1], X_model[f2], c=y.astype('category').cat.codes if hasattr(y, "cat") else y,
                             edgecolor="k")
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title("2D Decision Boundary")
        st.pyplot(fig)
    else:
        st.info("Select two different features for 2D plot.")
else:
    st.info("Dataset must have at least 2 features for 2D visualization.")

# -------------------------------
# 3D Decision Boundary
# -------------------------------
st.subheader("🌐 3D Decision Boundary")
if X.shape[1] >= 3:
    f3x = st.selectbox("Feature X", X.columns, key="3dx")
    f3y = st.selectbox("Feature Y", X.columns, index=1, key="3dy")
    f3z = st.selectbox("Feature Z", X.columns, index=2, key="3dz")

    X_vis3d = X_train[[f3x, f3y, f3z]]
    clf3d = VotingClassifier([("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard")
    clf3d.fit(X_vis3d, y_train)

    x_min, x_max = X_vis3d[f3x].min() - 1, X_vis3d[f3x].max() + 1
    y_min, y_max = X_vis3d[f3y].min() - 1, X_vis3d[f3y].max() + 1
    z_min, z_max = X_vis3d[f3z].min() - 1, X_vis3d[f3z].max() + 1

    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20),
        np.linspace(z_min, z_max, 20)
    )

    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = clf3d.predict(grid).reshape(xx.shape)

    scatter = go.Scatter3d(
        x=X_vis3d[f3x], y=X_vis3d[f3y], z=X_vis3d[f3z],
        mode='markers',
        marker=dict(size=5, color=y_train.astype(str)),
        name='Train Points'
    )

    surface = go.Volume(
        x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
        value=Z.flatten(),
        isomin=0, isomax=len(np.unique(y))-1,
        opacity=0.1,
        surface_count=len(np.unique(y)),
        colorscale='Viridis'
    )

    fig = go.Figure(data=[scatter, surface])
    fig.update_layout(scene=dict(
        xaxis_title=f3x, yaxis_title=f3y, zaxis_title=f3z
    ), height=700)
    st.plotly_chart(fig)
else:
    st.info("Dataset must have at least 3 features for 3D visualization.")

# Footer
st.markdown("---")
st.markdown("💡 Tip: Use GridSearchCV for automatic RF tuning, export/import model with Pickle, and visualize 2D/3D decision boundaries for interactive exploration.")




