
# app.py
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

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
st.title("🗳️ Voting Classifier Playground with 2D/3D Visualization")

# File Upload or Example
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
use_example = st.checkbox("Use Example Dataset (example_data.csv)")

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

# Train-Test Split & Standardization
st.subheader("Train/Test Split & Scaling")
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.number_input("Random Seed", 0, 9999, 42)
scale_features = st.checkbox("Standardize features", value=True)

if scale_features:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
else:
    X_scaled = X.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(pd.unique(y))>1 else None
)

# -------------------------------
# GridSearchCV for RandomForest
# -------------------------------
st.subheader("⚙️ Random Forest Hyperparameter Tuning")
n_estimators_list = st.multiselect("n_estimators", [50, 100, 200], default=[100])
max_depth_list = st.multiselect("max_depth", [None, 5, 10, 20], default=[None, 10])

run_grid = st.button("Run GridSearchCV for Random Forest")
if run_grid:
    rf = RandomForestClassifier(random_state=random_state)
    param_grid = {"n_estimators": n_estimators_list, "max_depth": max_depth_list}
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    st.success(f"Best Parameters: {grid.best_params_}")
    best_rf = grid.best_estimator_
else:
    best_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state)

# -------------------------------
# Voting Classifier
# -------------------------------
st.subheader("Train Voting Classifier")
lr = LogisticRegression(max_iter=1000, random_state=random_state)
svm = SVC(probability=False, random_state=random_state)

voting_clf = VotingClassifier(
    estimators=[("lr", lr), ("svm", svm), ("rf", best_rf)], voting="hard"
)

train_button = st.button("Train Voting Classifier")
if train_button:
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    st.subheader("📊 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Save model
    joblib.dump({"model": voting_clf, "scaler": scaler}, "voting_classifier.pkl")
    st.success("✅ Model trained and saved as `voting_classifier.pkl`")

# -------------------------------
# Load Saved Model
# -------------------------------
load_button = st.button("Load Existing Model")
if load_button:
    try:
        saved = joblib.load("voting_classifier.pkl")
        loaded_model = saved["model"]
        loaded_scaler = saved["scaler"]
        X_test_scaled = loaded_scaler.transform(X_test) if loaded_scaler else X_test
        y_pred_loaded = loaded_model.predict(X_test_scaled)
        st.success("✅ Loaded saved model successfully!")
        st.write("**Accuracy (Loaded Model):**", accuracy_score(y_test, y_pred_loaded))
    except FileNotFoundError:
        st.error("No saved model found. Train one first!")

# -------------------------------
# Feature Selection for Boundary Plots
# -------------------------------
st.subheader("Decision Boundary Visualization")
all_features = list(X.columns)
f1 = st.selectbox("Feature 1", all_features, index=0)
f2 = st.selectbox("Feature 2", all_features, index=1 if len(all_features)>1 else 0)
f3 = None
if len(all_features) >= 3:
    f3 = st.selectbox("Feature 3 (for 3D)", all_features, index=2)

# -------------------------------
# 2D / 3D Decision Boundary
# -------------------------------
viz_mode = st.radio("Plot Type", ["2D", "3D"])

if viz_mode=="2D":
    X_plot = X_scaled[[f1,f2]].values
    x_min, x_max = X_plot[:,0].min()-1, X_plot[:,0].max()+1
    y_min, y_max = X_plot[:,1].min()-1, X_plot[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    clf2 = VotingClassifier([("lr", lr),("svm", svm),("rf", best_rf)], voting="hard")
    clf2.fit(X_scaled[[f1,f2]].values, y)
    Z = clf2.predict(grid).reshape(xx.shape)
    fig2, ax = plt.subplots(figsize=(6,5))
    ax.contourf(xx,yy,Z, alpha=0.3)
    ax.scatter(X_plot[:,0], X_plot[:,1], c=y, edgecolor="k", s=30)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    st.pyplot(fig2)

elif viz_mode=="3D" and f3:
    X_plot = X_scaled[[f1,f2,f3]].values
    clf3 = VotingClassifier([("lr", lr),("svm", svm),("rf", best_rf)], voting="hard")
    clf3.fit(X_plot, y)
    x_min, x_max = X_plot[:,0].min()-1, X_plot[:,0].max()+1
    y_min, y_max = X_plot[:,1].min()-1, X_plot[:,1].max()+1
    z_min, z_max = X_plot[:,2].min()-1, X_plot[:,2].max()+1
    xx,yy,zz = np.meshgrid(np.linspace(x_min,x_max,20),
                            np.linspace(y_min,y_max,20),
                            np.linspace(z_min,z_max,20))
    grid3 = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z3 = clf3.predict(grid3)
    sample_idx = np.random.choice(len(Z3), min(5000,len(Z3)), replace=False)
    fig3 = go.Figure(data=[go.Scatter3d(
        x=grid3[sample_idx,0],
        y=grid3[sample_idx,1],
        z=grid3[sample_idx,2],
        mode='markers',
        marker=dict(size=2,color=Z3[sample_idx], colorscale='Viridis', opacity=0.3)
    )])
    fig3.add_trace(go.Scatter3d(
        x=X_plot[:,0], y=X_plot[:,1], z=X_plot[:,2],
        mode='markers',
        marker=dict(size=5, color=y, symbol='circle', line=dict(color='black', width=1))
    ))
    fig3.update_layout(scene=dict(xaxis_title=f1, yaxis_title=f2, zaxis_title=f3))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("3D visualization requires at least 3 features in dataset.")

# -------------------------------
# Feature Importances & RF Tree
# -------------------------------
st.subheader("Random Forest Feature Importances & Tree")

if best_rf:
    # Feature Importances
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances")
    st.pyplot(fig)

    # Decision Tree Plot (first tree of RF)
    fig_tree, ax_tree = plt.subplots(figsize=(12,8))
    plot_tree(best_rf.estimators_[0], feature_names=X.columns, filled=True, rounded=True)
    st.pyplot(fig_tree)

st.markdown("---")
st.markdown("💡 Tip: Toggle 2D/3D plots, tune RF parameters with GridSearchCV, and save/load your model for experimentation.")



