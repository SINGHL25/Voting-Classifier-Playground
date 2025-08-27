
# app.py
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

# -------------------------------
# Sidebar: Model Info
# -------------------------------
st.sidebar.title("ℹ️ About this Playground")
st.sidebar.markdown("""
This **ML Playground** lets you experiment with different classifiers:

- **Logistic Regression** – linear classifier
- **SVM** – good for non-linear boundaries
- **Random Forest** – ensemble of decision trees
- **Voting Classifier (hard)** – combines LR, SVM, RF
- **AdaBoost / Bagging / Gradient Boosting** – advanced ensembles

**Features:**
- Pick dataset or upload CSV
- Tune hyperparameters
- Train/test split & scaling
- See accuracy, classification report, confusion matrix
- 2D/3D decision boundary visualization
- Export/import trained model (pickle)
""")

# -------------------------------
# Main App
# -------------------------------
st.title("🖥️ ML Playground: Classifiers & Voting")

# --- Dataset ---
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

target_col = st.selectbox("Select Target Column", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Train/Test Split ---
st.sidebar.header("Split & Scaling")
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.sidebar.number_input("Random Seed", 0, 9999, 42)
scale_features = st.sidebar.checkbox("Standardize Features", True)

if scale_features:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
)

# --- Classifier Selection ---
st.sidebar.header("Select Classifier")
classifier_name = st.sidebar.selectbox(
    "Classifier",
    ["Logistic Regression", "SVM", "Random Forest", "Voting Classifier",
     "AdaBoost", "Bagging", "Gradient Boosting"]
)

# --- Hyperparameters ---
st.sidebar.header("Hyperparameters")
params = {}
if classifier_name in ["Random Forest", "Voting Classifier", "AdaBoost", "Bagging", "Gradient Boosting"]:
    params['n_estimators'] = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
if classifier_name in ["Random Forest", "Voting Classifier", "Gradient Boosting"]:
    params['max_depth'] = st.sidebar.slider("max_depth (None=0)", 0, 20, 5)
if classifier_name in ["Logistic Regression"]:
    params['C'] = st.sidebar.number_input("C (Inverse Regularization)", 0.01, 10.0, 1.0)
if classifier_name in ["SVM"]:
    params['C'] = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, 1.0)
    params['kernel'] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

# --- Initialize Classifier ---
def get_classifier(name):
    if name == "Logistic Regression":
        return LogisticRegression(C=params.get('C',1.0), max_iter=1000, random_state=random_state)
    elif name == "SVM":
        return SVC(C=params.get('C',1.0), kernel=params.get('kernel','linear'), probability=True, random_state=random_state)
    elif name == "Random Forest":
        md = None if params.get('max_depth',0)==0 else params.get('max_depth')
        return RandomForestClassifier(n_estimators=params.get('n_estimators',100), max_depth=md, random_state=random_state)
    elif name == "Voting Classifier":
        rf_md = None if params.get('max_depth',0)==0 else params.get('max_depth')
        rf = RandomForestClassifier(n_estimators=params.get('n_estimators',100), max_depth=rf_md, random_state=random_state)
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        svm = SVC(probability=True, random_state=random_state)
        return VotingClassifier(estimators=[('lr',lr),('svm',svm),('rf',rf)], voting='hard')
    elif name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=params.get('n_estimators',100), random_state=random_state)
    elif name == "Bagging":
        return BaggingClassifier(n_estimators=params.get('n_estimators',100), random_state=random_state)
    elif name == "Gradient Boosting":
        md = None if params.get('max_depth',0)==0 else params.get('max_depth')
        return GradientBoostingClassifier(n_estimators=params.get('n_estimators',100), max_depth=md, random_state=random_state)
    else:
        return LogisticRegression()

clf = get_classifier(classifier_name)

# --- Train & Evaluate ---
if st.button("Train & Evaluate"):
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.subheader("📊 Model Evaluation")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save model
        joblib.dump(clf, "ml_playground_model.pkl")
        st.success("✅ Model trained and saved as `ml_playground_model.pkl`")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(np.unique(y))
        ax.set_yticklabels(np.unique(y))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
        st.pyplot(fig)

        # 2D Decision Boundary (if 2 features)
        if X.shape[1] >= 2:
            f1, f2 = 0,1
            X_plot = X_train[:,[f1,f2]]
            y_plot = y_train
            x_min, x_max = X_plot[:,0].min()-1, X_plot[:,0].max()+1
            y_min, y_max = X_plot[:,1].min()-1, X_plot[:,1].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
            try:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                fig2, ax2 = plt.subplots()
                ax2.contourf(xx,yy,Z,alpha=0.3)
                ax2.scatter(X_plot[:,0],X_plot[:,1],c=y_plot,edgecolor='k')
                ax2.set_xlabel(df.columns[f1])
                ax2.set_ylabel(df.columns[f2])
                ax2.set_title("2D Decision Boundary")
                st.pyplot(fig2)
            except:
                st.warning("Cannot plot 2D decision boundary for this classifier or dataset.")

    except Exception as e:
        st.error(f"Training failed: {e}")

# --- Load Model ---
if st.button("Load Existing Model"):
    try:
        loaded = joblib.load("ml_playground_model.pkl")
        st.success("✅ Loaded saved model!")
    except FileNotFoundError:
        st.error("No saved model found. Train first!")

st.markdown("---")
st.markdown("💡 Tip: Try different classifiers, toggle hyperparameters, and upload your own dataset for testing.")




