"""
Optimized Credit Scoring with XGBoost (Windows-friendly)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)
from xgboost import XGBClassifier
import joblib

# -----------------------
# CONFIG
# -----------------------
SEED = 42
DATA_FILE = "cs-training.csv"
OUT_DIR = "output_model"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEARCH_ITERS = 20   # số lần thử tham số
CV_FOLDS = 4

# -----------------------
# 1) Load data
# -----------------------
print("Loading data:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "SeriousDlqin2yrs" in df.columns:
    df = df.rename(columns={"SeriousDlqin2yrs": "TARGET"})
elif "TARGET" not in df.columns:
    raise ValueError("Không tìm thấy cột target!")

print("Shape:", df.shape)

# -----------------------
# 2) Cleaning
# -----------------------
if "age" in df.columns:
    df["age"] = df["age"].clip(18, 100)

df = df.drop_duplicates()

# -----------------------
# 3) Split
# -----------------------
X = df.drop(columns=["TARGET"])
y = df["TARGET"].astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=SEED
)

# -----------------------
# 4) Preprocessing
# -----------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
numeric_pipeline.fit(X_train)

X_train_proc = numeric_pipeline.transform(X_train)
X_val_proc   = numeric_pipeline.transform(X_val)
X_test_proc  = numeric_pipeline.transform(X_test)
X_train_full_proc = numeric_pipeline.transform(X_train_full)

# -----------------------
# 5) Class imbalance
# -----------------------
n_pos = y_train_full.sum()
n_neg = len(y_train_full) - n_pos
scale_pos_weight = n_neg / max(1, n_pos)
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

# -----------------------
# 6) Baseline
# -----------------------
base_clf = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    n_jobs=1,   # Windows-safe
    random_state=SEED
)
base_clf.fit(X_train_proc, y_train)
print("Baseline ROC-AUC:", roc_auc_score(y_val, base_clf.predict_proba(X_val_proc)[:,1]))

# -----------------------
# 7) RandomizedSearchCV
# -----------------------
param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10]
}

est = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    n_jobs=1,  # tránh crash
    random_state=SEED
)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
rand = RandomizedSearchCV(
    estimator=est,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITERS,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,   # ⚡ sửa: chạy tuần tự để tránh lỗi _posixsubprocess
    verbose=2,
    random_state=SEED
)
rand.fit(X_train_full_proc, y_train_full)

print("Best CV AUC:", rand.best_score_)
print("Best params:", rand.best_params_)

best_params = rand.best_params_
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": 1,
    "random_state": SEED
})

# -----------------------
# 8) Final model
# -----------------------
final_clf = XGBClassifier(**best_params)
final_clf.fit(X_train_full_proc, y_train_full, eval_set=[(X_val_proc, y_val)], verbose=False)

# -----------------------
# 9) Precision optimization
# -----------------------
y_proba = final_clf.predict_proba(X_test_proc)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

best_thresh = thresholds[np.argmax(precisions * (recalls > 0.2))]
print(f"Chosen threshold = {best_thresh:.3f}")

y_pred = (y_proba >= best_thresh).astype(int)

print("\nFinal Test ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------
# 10) Save
# -----------------------
joblib.dump(final_clf, os.path.join(OUT_DIR, "xgb_final_model.joblib"))
joblib.dump(numeric_pipeline, os.path.join(OUT_DIR, "preprocessor.joblib"))
print("Models saved to", OUT_DIR)
