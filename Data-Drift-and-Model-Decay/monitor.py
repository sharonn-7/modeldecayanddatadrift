import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
import logging
import os
import argparse
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_psi(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    bins_edges = np.linspace(expected.min(), expected.max() + 1e-5, bins + 1)
    expected_counts, _ = np.histogram(expected, bins=bins_edges)
    actual_counts,   _ = np.histogram(actual,   bins=bins_edges)
    expected_pct = np.where(expected_counts / len(expected) == 0, 1e-6, expected_counts / len(expected))
    actual_pct   = np.where(actual_counts   / len(actual)   == 0, 1e-6, actual_counts   / len(actual))
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

def detect_drift_psi(baseline_df, new_df):
    results = []
    for col in baseline_df.select_dtypes(include=[np.number]).columns:
        if col in new_df.columns:
            expected = baseline_df[col].dropna().values
            actual   = new_df[col].dropna().values
            if len(expected) > 0 and len(actual) > 0:
                results.append({'Feature': col, 'PSI': calculate_psi(expected, actual)})
    df_report = pd.DataFrame(results)
    if not df_report.empty:
        df_report = df_report.sort_values(by='PSI', ascending=False)
    return df_report

def run_monitoring(model_path='model.pkl', scaler_path='scaler.pkl',
                   imputer_path='imputer.pkl', baseline_path='baseline.pkl',
                   new_data_path='new_data.csv', target_col='default.payment.next.month'):

    logging.info("Loading artifacts...")
    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    imputer  = joblib.load(imputer_path)
    baseline = joblib.load(baseline_path)

    baseline_df      = baseline['X_train']
    baseline_roc_auc = baseline.get('baseline_roc_auc', 0.75)

    logging.info(f"Loading new data from {new_data_path}")
    new_df = pd.read_csv(new_data_path)
    if 'ID' in new_df.columns:
        new_df = new_df.drop(columns=['ID'])

    y_true = None
    if target_col in new_df.columns:
        y_true = new_df[target_col]
        X_new  = new_df.drop(columns=[target_col])
    else:
        X_new  = new_df.copy()

    logging.info("Computing PSI Drift...")
    drift_report = detect_drift_psi(baseline_df, X_new)
    drift_score  = drift_report['PSI'].mean() if not drift_report.empty else 0.0

    logging.info("Preprocessing new data...")
    X_new_imputed   = imputer.transform(X_new)
    X_new_scaled    = scaler.transform(X_new_imputed)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X_new.columns)

    logging.info("Predicting...")
    y_prob = model.predict_proba(X_new_scaled_df)[:, 1]

    current_roc_auc = None
    roc_auc_drop    = 0.0
    if y_true is not None:
        current_roc_auc = roc_auc_score(y_true, y_prob)
        roc_auc_drop    = max(0, baseline_roc_auc - current_roc_auc)

    drift_impact_score = (0.6 * drift_score) + (0.4 * roc_auc_drop) if current_roc_auc else drift_score

    if drift_score > 0.25:
        trend_status = "At Risk — high drift detected"
    elif drift_score > 0.10:
        trend_status = "Caution — moderate drift"
    else:
        trend_status = "Stable"

    alert_triggered = drift_impact_score > 0.05

    logging.info(f"Drift Score (Avg PSI) : {drift_score:.4f}")
    if current_roc_auc is not None:
        logging.info(f"ROC-AUC              : {current_roc_auc:.4f}  (baseline: {baseline_roc_auc:.4f}, drop: {roc_auc_drop:.4f})")
    logging.info(f"Drift Impact Score   : {drift_impact_score:.4f}")
    logging.info(f"Trend Status         : {trend_status}")
    logging.info(f"Alert Triggered      : {'Yes' if alert_triggered else 'No'}")

    return {
        'drift_score':        drift_score,
        'drift_report':       drift_report,
        'current_roc_auc':    current_roc_auc,
        'roc_auc_drop':       roc_auc_drop,
        'drift_impact_score': drift_impact_score,
        'trend_status':       trend_status,
        'alert_triggered':    alert_triggered
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run monitoring pipeline")
    parser.add_argument('--new_data', default='new_data.csv', help='Path to new data file')
    args = parser.parse_args()
    if os.path.exists(args.new_data):
        run_monitoring(new_data_path=args.new_data)
    else:
        logging.error(f"New data file {args.new_data} not found.")
