import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             accuracy_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------
# DATA PREPARATION
# -----------------------------------
df = pd.read_csv('UCI_Credit_Card.csv')
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

target_col = 'default.payment.next.month'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

imputer = SimpleImputer(strategy='mean')
X_train_sc = StandardScaler().fit_transform(imputer.fit_transform(X_train))
X_test_sc  = StandardScaler().fit_transform(imputer.transform(X_test))

# refit scaler properly
from sklearn.preprocessing import StandardScaler as SS
sc = SS()
X_train_sc = sc.fit_transform(imputer.fit_transform(X_train))
X_test_sc  = sc.transform(imputer.transform(X_test))

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

# -----------------------------------
# EVALUATION FUNCTION
# -----------------------------------
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy_score(y_te, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_te, y_prob):.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_te, y_pred,
                                   target_names=['No Default', 'Default'],
                                   zero_division=0)
    for line in report.splitlines():
        print(f"    {line}")

    return {
        'accuracy':  accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall':    recall_score(y_te, y_pred, zero_division=0),
        'f1':        f1_score(y_te, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_te, y_prob),
    }

# -----------------------------------
# MODEL 1 — Random Forest
# -----------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = evaluate("Random Forest", rf, X_train_sc, y_train, X_test_sc, y_test)

# -----------------------------------
# MODEL 2 — XGBoost
# -----------------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=spw,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_scores = evaluate(f"XGBoost (scale_pos_weight={spw:.2f})", xgb, X_train_sc, y_train, X_test_sc, y_test)

# -----------------------------------
# SIDE-BY-SIDE SUMMARY
# -----------------------------------
print(f"\n\n{'='*55}")
print("  COMPARISON SUMMARY")
print(f"{'='*55}")
print(f"  {'Metric':<12} {'Random Forest':>15} {'XGBoost':>12} {'Winner':>10}")
print(f"  {'-'*50}")

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    rf_val  = rf_scores[metric]
    xgb_val = xgb_scores[metric]
    winner  = 'XGBoost' if xgb_val > rf_val else 'RF'
    diff    = abs(xgb_val - rf_val)
    print(f"  {metric.upper():<12} {rf_val:>15.4f} {xgb_val:>12.4f} {winner:>10}  (+{diff:.4f})")

print(f"\n  Class imbalance : {spw:.2f}x  ({pos:,} defaults / {neg:,} non-defaults)")
print(f"  Test set size   : {len(y_test):,} rows")
