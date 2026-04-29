import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, roc_curve

# -----------------------------------
# ANALYTICS UTILITIES
# -----------------------------------
def calculate_psi(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0: return 0.0
    bins_edges = np.linspace(expected.min(), expected.max() + 1e-5, bins + 1)
    e_pct = np.clip(np.histogram(expected, bins=bins_edges)[0] / len(expected), 1e-6, None)
    a_pct = np.clip(np.histogram(actual,   bins=bins_edges)[0] / len(actual),   1e-6, None)
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

def calculate_kl_divergence(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0: return 0.0
    hist_e = np.clip(np.histogram(expected, bins=bins, density=True)[0], 1e-6, None)
    hist_a = np.clip(np.histogram(actual,   bins=bins, density=True)[0], 1e-6, None)
    return entropy(hist_a, hist_e)

def calculate_js_divergence(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0: return 0.0
    hist_e = np.clip(np.histogram(expected, bins=bins, density=True)[0], 1e-6, None)
    hist_a = np.clip(np.histogram(actual,   bins=bins, density=True)[0], 1e-6, None)
    m = 0.5 * (hist_e + hist_a)
    return 0.5 * entropy(hist_e, m) + 0.5 * entropy(hist_a, m)

def get_unified_drift(expected, actual):
    psi = calculate_psi(expected, actual)
    kl  = calculate_kl_divergence(expected, actual)
    js  = calculate_js_divergence(expected, actual)
    return (0.5 * psi) + (0.25 * kl) + (0.25 * js), psi, kl, js

# -----------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------
st.set_page_config(page_title="Adaptive ML Observability", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #020617 0%, #0f172a 100%); color: #f8fafc; }
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1.5rem; border-radius: 20px; backdrop-filter: blur(12px);
    }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("Adaptive ML Observability Framework")
st.caption("Multi-Metric Drift Detection | Dynamic Baseline | ROC-AUC Monitoring")

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.header("Configuration")
    alpha       = st.slider("Drift Influence (alpha)", 0.0, 1.0, 0.6)
    threshold   = st.number_input("Alert Threshold", value=0.1, step=0.01)
    window_size = st.number_input("Rolling Window Size", value=5, min_value=1)
    target_col  = st.text_input("Target Column Name", value="default.payment.next.month")

    st.divider()
    st.header("Upload Artifacts")
    model_file    = st.file_uploader("Model (.pkl)",    type=['pkl'])
    scaler_file   = st.file_uploader("Scaler (.pkl)",   type=['pkl'])
    imputer_file  = st.file_uploader("Imputer (.pkl)",  type=['pkl'])
    baseline_file = st.file_uploader("Baseline (.pkl)", type=['pkl'])
    new_data_file = st.file_uploader("New Data (.csv)", type=['csv'])

if not (model_file and baseline_file and new_data_file):
    st.info("Upload Model, Baseline, and New Data to begin monitoring. Scaler and Imputer are optional but required for accurate ROC-AUC.")
    st.stop()

# -----------------------------------
# LOAD ARTIFACTS
# -----------------------------------
@st.cache_resource
def load_artifacts(_model_f, _baseline_f, _scaler_f, _imputer_f):
    model    = joblib.load(_model_f)
    baseline = joblib.load(_baseline_f)
    scaler   = joblib.load(_scaler_f)  if _scaler_f  else None
    imputer  = joblib.load(_imputer_f) if _imputer_f else None
    return model, baseline, scaler, imputer

try:
    model, baseline_obj, scaler, imputer = load_artifacts(
        model_file, baseline_file, scaler_file, imputer_file)
    baseline_df      = baseline_obj['X_train']
    baseline_roc_auc = baseline_obj.get('baseline_roc_auc', 0.75)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

if 'rolling_buffer' not in st.session_state:
    st.session_state.rolling_buffer = [baseline_df]

# -----------------------------------
# PROCESS INCOMING DATA
# -----------------------------------
new_df = pd.read_csv(new_data_file)

if target_col and target_col in new_df.columns:
    y_true = new_df[target_col]
    X_new  = new_df.drop(columns=[target_col])
else:
    X_new  = new_df.copy()
    y_true = None
    st.warning("Target column not found. ROC-AUC will be unavailable.")

if 'ID' in X_new.columns:
    X_new = X_new.drop(columns=['ID'])

current_reference = pd.concat(st.session_state.rolling_buffer[-window_size:])

# -----------------------------------
# DRIFT CALCULATION (on raw features)
# -----------------------------------
feature_importance = {}
if hasattr(model, 'feature_importances_'):
    fi_cols = [c for c in X_new.columns if c in baseline_df.columns]
    feature_importance = dict(zip(fi_cols, model.feature_importances_[:len(fi_cols)]))
elif hasattr(model, 'coef_'):
    fi_cols = [c for c in X_new.columns if c in baseline_df.columns]
    feature_importance = dict(zip(fi_cols, np.abs(model.coef_[0])[:len(fi_cols)]))

drift_results = []
for col in X_new.select_dtypes(include=[np.number]).columns:
    if col in current_reference.columns:
        u_score, psi, kl, js = get_unified_drift(
            current_reference[col].dropna().values,
            X_new[col].dropna().values
        )
        importance = feature_importance.get(col, 1.0)
        drift_results.append({
            'Feature': col, 'Unified Drift': u_score,
            'PSI': psi, 'KL': kl, 'JS': js,
            'Importance': importance,
            'Impact Rank': u_score * importance
        })

drift_summary_df  = pd.DataFrame(drift_results).sort_values(by='Impact Rank', ascending=False)
avg_unified_drift = drift_summary_df['Unified Drift'].mean() if not drift_summary_df.empty else 0.0

# -----------------------------------
# ROC-AUC (preprocess if scaler/imputer available)
# -----------------------------------
current_roc_auc = None
roc_auc_drop    = 0.0
fpr = tpr = None

try:
    if imputer and scaler:
        X_proc = pd.DataFrame(
            scaler.transform(imputer.transform(X_new)),
            columns=X_new.columns
        )
    else:
        X_proc = X_new

    y_prob = model.predict_proba(X_proc)[:, 1]

    if y_true is not None:
        current_roc_auc = roc_auc_score(y_true, y_prob)
        roc_auc_drop    = max(0.0, baseline_roc_auc - current_roc_auc)
        fpr, tpr, _     = roc_curve(y_true, y_prob)
except Exception as e:
    st.error(f"Prediction Error: {e}")

# -----------------------------------
# DRIFT IMPACT SCORE
# -----------------------------------
drift_impact_score = (alpha * avg_unified_drift) + ((1 - alpha) * roc_auc_drop)

# Health logic — purely off drift_impact_score so it always changes
if current_roc_auc is not None and roc_auc_drop > 0.02:
    health_label = "Decline"
    health_delta = f"-{roc_auc_drop:.4f} ROC-AUC"
elif drift_impact_score > threshold:
    health_label = "At Risk"
    health_delta = f"score {drift_impact_score:.4f}"
elif drift_impact_score > threshold * 0.5:
    health_label = "Caution"
    health_delta = f"score {drift_impact_score:.4f}"
else:
    health_label = "Stable"
    health_delta = f"score {drift_impact_score:.4f}"

# -----------------------------------
# METRIC CARDS
# -----------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Unified Drift Score", f"{avg_unified_drift:.4f}")
m2.metric("Drift Impact Score",  f"{drift_impact_score:.4f}")
m3.metric("ROC-AUC",
          f"{current_roc_auc:.4f}" if current_roc_auc is not None else "N/A",
          delta=f"baseline {baseline_roc_auc:.4f}" if current_roc_auc is not None else "upload scaler & imputer")
m4.metric("Model Health", health_label, delta=health_delta)

st.divider()

# -----------------------------------
# ALERTS
# -----------------------------------
if drift_impact_score > threshold:
    st.error(f"CRITICAL ALERT: Drift Impact Score ({drift_impact_score:.4f}) exceeds threshold ({threshold}). Retraining recommended.")
elif health_label in ("Decline", "At Risk"):
    st.warning("DECAY WARNING: Model performance or drift has crossed safe limits.")
else:
    st.success("System Status: Stable. No significant impact detected.")

# -----------------------------------
# LAYOUT
# -----------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Feature-Level Drift Analysis")
    st.dataframe(
        drift_summary_df.head(10).style.background_gradient(subset=['Unified Drift'], cmap='OrRd'),
        use_container_width=True
    )

with col_right:
    st.subheader("ROC Curve")
    if fpr is not None and tpr is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.style.use('dark_background')
        ax.plot(fpr, tpr, color='#38bdf8', lw=2, label=f"AUC = {current_roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', lw=1, label="Random")
        ax.set_xlabel("False Positive Rate", color='#94a3b8')
        ax.set_ylabel("True Positive Rate",  color='#94a3b8')
        ax.set_title("ROC Curve", color='#f8fafc')
        ax.tick_params(colors='#94a3b8')
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        ax.legend(facecolor='#1e293b', labelcolor='#f8fafc')
        st.pyplot(fig)
    else:
        st.info("Upload scaler.pkl + imputer.pkl and set a target column to see the ROC curve.")

    st.subheader("Drift Metric Breakdown")
    st.bar_chart({'PSI': drift_summary_df['PSI'].mean(),
                  'KL':  drift_summary_df['KL'].mean(),
                  'JS':  drift_summary_df['JS'].mean()})

    st.info("""
    **Metrics:**
    - **ROC-AUC**: Probability model ranks a defaulter above a non-defaulter.
    - **PSI**: Distribution stability of each feature.
    - **Impact Rank**: Drift x Feature Importance.
    """)

# -----------------------------------
# ROLLING BASELINE
# -----------------------------------
if st.button("Commit to Rolling Baseline"):
    st.session_state.rolling_buffer.append(X_new)
    if len(st.session_state.rolling_buffer) > window_size:
        st.session_state.rolling_buffer.pop(0)
    st.toast("Current batch added to reference buffer!")
