import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import joblib
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save(filepath='UCI_Credit_Card.csv', target_col='default.payment.next.month'):
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logging.info("Preprocessing data...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled  = scaler.transform(X_test_imputed)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    logging.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_f1 = f1_score(y_test, rf.predict(X_test_scaled), zero_division=0)

    logging.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_f1 = f1_score(y_test, xgb.predict(X_test_scaled), zero_division=0)

    if xgb_f1 >= rf_f1:
        model, model_name = xgb, "XGBoost"
    else:
        model, model_name = rf, "Random Forest"

    logging.info(f"RF F1={rf_f1:.4f}  XGB F1={xgb_f1:.4f}  -> Selected: {model_name}")

    logging.info("Evaluating selected model...")
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    baseline_roc_auc = roc_auc_score(y_test, y_prob)

    baseline_data = {
        'X_train': X_train.copy(),
        'baseline_roc_auc': baseline_roc_auc
    }

    logging.info(f"Baseline ROC-AUC ({model_name}): {baseline_roc_auc:.4f}")
    logging.info("Saving artifacts...")
    joblib.dump(model,         'model.pkl')
    joblib.dump(scaler,        'scaler.pkl')
    joblib.dump(imputer,       'imputer.pkl')
    joblib.dump(baseline_data, 'baseline.pkl')
    logging.info("Training complete. Artifacts saved: model.pkl, scaler.pkl, imputer.pkl, baseline.pkl")

if __name__ == '__main__':
    train_and_save()
