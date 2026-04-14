import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

# -----------------------------
# 0) Load data
# -----------------------------
CSV_PATH = r"data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"

df = pd.read_csv(CSV_PATH)

# Drop ID column
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Target Yes/No -> 1/0
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = df[TARGET].values
X_df = df.drop(columns=[TARGET])

# Identify numeric vs categorical
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_df.columns if c not in num_cols]

# -----------------------------
# 1) Split (train/val/test)
# -----------------------------
X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X_df, y, test_size=0.30, random_state=42, stratify=y
)
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# -----------------------------
# 2) Preprocess
#   - Numeric: median imputation
#   - Categorical: most-frequent imputation + ordinal encoding
# -----------------------------
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

X_train = preprocess.fit_transform(X_train_df)
X_val = preprocess.transform(X_val_df)
X_test = preprocess.transform(X_test_df)

# -----------------------------
# 3) XGBoost models: L2 vs L1
# -----------------------------

pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = neg / max(pos, 1)

# L2-regularized model (Ridge)
model_l2 = XGBClassifier(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,   # L2 on
    reg_alpha=0.0,    # L1 off
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50
)

# L1-regularized model (Lasso)
model_l1 = XGBClassifier(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.0,   # L2 off
    reg_alpha=1.0,    # L1 on
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50
)

model_l2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

model_l1.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"L2 best iteration: {model_l2.best_iteration}")
print(f"L1 best iteration: {model_l1.best_iteration}")


# -----------------------------
# 4) Benchmarking comparison: L2 vs L1
# -----------------------------

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

def evaluate_model(name, model, X_test, y_test, threshold=0.3):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)
    f1 = f1_score(y_test, y_preds)

    print(f"\n=== {name} RESULTS ===")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"Brier Score:  {brier:.4f}")
    print(f"F1 Score:     {f1:.4f}")

    return y_preds

test_preds_l2 = evaluate_model("XGBoost L2", model_l2, X_test, y_test)
test_preds_l1 = evaluate_model("XGBoost L1", model_l1, X_test, y_test)

# -----------------------------
# 5) Confusion Matrix
# -----------------------------

print("\n=== CONFUSION MATRIX (L2) ===")
print(confusion_matrix(y_test, test_preds_l2))

print("\n=== CONFUSION MATRIX (L1) ===")
print(confusion_matrix(y_test, test_preds_l1))

# -----------------------------
# 6) Additional Metrics
# -----------------------------

print("\n=== ADDITIONAL METRICS (L2) ===")
precision_l2 = precision_score(y_test, test_preds_l2)
recall_l2 = recall_score(y_test, test_preds_l2)

print(f"Precision: {precision_l2:.4f}")
print(f"Recall:    {recall_l2:.4f}")

print("\n=== ADDITIONAL METRICS (L1) ===")
precision_l1 = precision_score(y_test, test_preds_l1)
recall_l1 = recall_score(y_test, test_preds_l1)

print(f"Precision: {precision_l1:.4f}")
print(f"Recall:    {recall_l1:.4f}")

# -----------------------------
# 7) Feature importance
# -----------------------------

importance_df_l2 = pd.DataFrame({
    "Feature": num_cols + cat_cols,
    "Importance": model_l2.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df_l1 = pd.DataFrame({
    "Feature": num_cols + cat_cols,
    "Importance": model_l1.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== TOP 10 FEATURES: L2 ===")
print(importance_df_l2.head(10).to_string(index=False))

print("\n=== TOP 10 FEATURES: L1 ===")
print(importance_df_l1.head(10).to_string(index=False))