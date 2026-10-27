# Exported from the corresponding Jupyter notebook. Run from this folder.

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay, classification_report

df = pd.read_csv("data/bank-full.csv", sep=";")
print("Dataset dimensions:", df.shape)
display(df.head(10))
print("\nData types:")
display(df.dtypes.to_frame("dtype"))
print("\nDescriptive statistics:")
display(df.describe(include="all").T)


print("Missing values by column:")
display(df.isna().sum().sort_values(ascending=False).to_frame("missing_count"))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Target counts:")
display(df["y"].value_counts().to_frame("customers"))
print("Target percentages:")
display((df["y"].value_counts(normalize=True)*100).round(2).to_frame("percent"))


target_counts = df["y"].value_counts().reindex(["no","yes"])
plt.figure(figsize=(6,4))
plt.bar(target_counts.index, target_counts.values)
plt.title("Bank Term Deposit Subscription Distribution")
plt.xlabel("Subscription outcome")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,4))
df.groupby("job")["y"].apply(lambda s: (s=="yes").mean()).sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Subscription rate by job")
axes[0].set_xlabel("Subscription rate"); axes[0].set_ylabel("Job")
df.groupby("education")["y"].apply(lambda s: (s=="yes").mean()).sort_values().plot(kind="bar", ax=axes[1])
axes[1].set_title("Subscription rate by education")
axes[1].set_xlabel("Education"); axes[1].set_ylabel("Subscription rate")
plt.tight_layout(); plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
for label in ["no","yes"]:
    axes[0].hist(df.loc[df["y"]==label, "age"], bins=25, alpha=.55, label=label)
axes[0].set_title("Customer age by subscription outcome")
axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Number of customers"); axes[0].legend()
df.groupby("campaign")["y"].apply(lambda s: (s=="yes").mean()).head(15).plot(kind="bar", ax=axes[1])
axes[1].set_title("Subscription rate by number of contacts")
axes[1].set_xlabel("Contacts during this campaign"); axes[1].set_ylabel("Subscription rate")
plt.tight_layout(); plt.show()


target = "y"
drop_cols = ["y", "duration"]
X = df.drop(columns=drop_cols)
y = (df[target] == "yes").astype(int)
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))
print("Numerical predictors:", numeric_features)
print("Categorical predictors:", categorical_features)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
}
fitted = {}
predictions = {}
probabilities = {}
rows = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:,1]
    fitted[name] = pipe
    predictions[name] = pred
    probabilities[name] = prob
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })
results = pd.DataFrame(rows).set_index("Model").sort_values("F1-score", ascending=False)
display(results.round(4))


metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score"]
results[metrics_to_plot].plot(kind="bar", figsize=(11,5))
plt.title("Classification Model Performance Comparison")
plt.xlabel("Classifier"); plt.ylabel("Score (0 to 1)")
plt.xticks(rotation=20, ha="right"); plt.legend(loc="lower right")
plt.tight_layout(); plt.show()


fig, axes = plt.subplots(2, 2, figsize=(10,8))
for ax, (name, pred) in zip(axes.ravel(), predictions.items()):
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(name + " — Confusion Matrix")
    ax.set_xticks([0,1], ["Predicted no","Predicted yes"])
    ax.set_yticks([0,1], ["Actual no","Actual yes"])
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
plt.tight_layout(); plt.show()


plt.figure(figsize=(8,6))
for name, prob in probabilities.items():
    RocCurveDisplay.from_predictions(y_test, prob, name=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})", ax=plt.gca())
plt.title("ROC Curves for Bank Marketing Classifiers")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.tight_layout(); plt.show()


best_model_name = results["F1-score"].idxmax()
print("Best model by F1-score:", best_model_name)
print("\nClassification report for selected model:")
print(classification_report(y_test, predictions[best_model_name], target_names=["No subscription","Subscription"], zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions[best_model_name]))
