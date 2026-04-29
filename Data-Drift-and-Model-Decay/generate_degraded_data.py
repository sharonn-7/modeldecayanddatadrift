import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load original data and trained artifacts
df = pd.read_csv('UCI_Credit_Card.csv')
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

model   = joblib.load('model.pkl')
scaler  = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

target_col = 'default.payment.next.month'
X_orig = df.drop(columns=[target_col])
y_orig = df[target_col]

# --- Baseline ROC-AUC on original data ---
X_proc  = scaler.transform(imputer.transform(X_orig))
y_prob  = model.predict_proba(X_proc)[:, 1]
orig_auc = roc_auc_score(y_orig, y_prob)
print(f"Baseline ROC-AUC on original data : {orig_auc:.4f}")

# -------------------------------------------
# BUILD DEGRADED DATASET
# Three types of degradation applied together:
#
# 1. Heavy feature drift  — shift key feature distributions far from training
# 2. Concept drift        — corrupt the relationship between features & labels
#                           by flipping 30% of labels in the opposite direction
# 3. High noise           — add large Gaussian noise to all numeric columns
# -------------------------------------------
degraded = df.copy()

# 1. Heavy feature drift on the model's most influential features
degraded['LIMIT_BAL'] = degraded['LIMIT_BAL'] * 3.0    # triple credit limits
degraded['PAY_0']     = degraded['PAY_0']     + 3      # everyone 3 months more delayed
degraded['PAY_2']     = degraded['PAY_2']     + 3
degraded['PAY_3']     = degraded['PAY_3']     + 3
degraded['BILL_AMT1'] = degraded['BILL_AMT1'] * 2.5    # much higher bills
degraded['BILL_AMT2'] = degraded['BILL_AMT2'] * 2.5
degraded['PAY_AMT1']  = degraded['PAY_AMT1']  * 0.2   # much lower payments
degraded['PAY_AMT2']  = degraded['PAY_AMT2']  * 0.2
degraded['AGE']       = degraded['AGE']       + 10     # 10-year age shift

# 2. Concept drift — flip 30% of labels (high-confidence predictions reversed)
#    Targets in the wrong direction confuse the model's learned boundaries
flip_idx = df.sample(frac=0.30, random_state=42).index
degraded.loc[flip_idx, target_col] = 1 - degraded.loc[flip_idx, target_col]

# 3. Heavy noise on all numeric columns except target
num_cols = degraded.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != target_col]
degraded[num_cols] = degraded[num_cols] + np.random.normal(0, 0.5, degraded[num_cols].shape)

# --- Measure ROC-AUC on degraded data ---
X_deg  = degraded.drop(columns=[target_col])
y_deg  = degraded[target_col]
X_proc_deg = scaler.transform(imputer.transform(X_deg))
y_prob_deg = model.predict_proba(X_proc_deg)[:, 1]
deg_auc = roc_auc_score(y_deg, y_prob_deg)

print(f"ROC-AUC on degraded data          : {deg_auc:.4f}")
print(f"ROC-AUC drop                      : {orig_auc - deg_auc:+.4f}")
print(f"Model Health should show          : {'Decline' if orig_auc - deg_auc > 0.02 else 'At Risk'}")

# Save
degraded.to_csv('degraded_data.csv', index=False)
print(f"\ndegraded_data.csv saved — {len(degraded)} rows x {len(degraded.columns)} columns")
print("Upload this as 'New Data' in the dashboard to see Model Health change.")
