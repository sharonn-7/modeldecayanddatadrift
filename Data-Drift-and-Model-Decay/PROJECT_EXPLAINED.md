# Data Drift and Model Decay — Complete Project Documentation

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Workflow](#2-high-level-workflow)
3. [Directory Structure](#3-directory-structure)
4. [Step-by-Step Pipeline](#4-step-by-step-pipeline)
5. [File-by-File Deep Dive](#5-file-by-file-deep-dive)
6. [Dashboard — Charts & Graphs Explained](#6-dashboard--charts--graphs-explained)
7. [Key Concepts](#7-key-concepts)
8. [Datasets](#8-datasets)
9. [Model Evolution — Why XGBoost](#9-model-evolution--why-xgboost)
10. [Generated Artifacts](#10-generated-artifacts)
11. [Running the Project](#11-running-the-project)

---

## 1. What This Project Does

This is a **Machine Learning Observability System** for credit card default prediction. It monitors a trained ML model in production to detect two problems:

- **Data Drift** — when the statistical distribution of incoming data changes compared to what the model was trained on (e.g., credit limits inflate, payment patterns shift)
- **Model Decay** — when the model's prediction performance degrades over time, measured by ROC-AUC drop

The system has three monitoring interfaces:
- `train.py` — trains the model and saves all artifacts
- `monitor.py` — CLI script for automated/scheduled monitoring runs
- `app.py` — interactive Streamlit web dashboard with charts and alerts

---

## 2. High-Level Workflow

```
Step 1 — Generate test data with drift        (new.py / generate_degraded_data.py)
Step 2 — Train the XGBoost model              (train.py)
Step 3 — Monitor for drift & decay            (monitor.py  OR  streamlit run app.py)
Step 4 — Compare models for best performance  (compare_models.py)
```

---

## 3. Directory Structure

```
Data-Drift-and-Model-Decay/
├── train.py                        Train XGBoost, save all artifacts
├── monitor.py                      CLI monitoring pipeline
├── app.py                          Streamlit interactive dashboard
├── new.py                          Generate mildly drifted test data
├── generate_degraded_data.py       Generate heavily degraded data (ROC-AUC drops)
├── compare_models.py               Compare RF vs XGBoost vs LightGBM with all techniques
├── UCI_Credit_Card.csv             Original dataset — 30,000 rows (baseline)
├── UCI_Credit_Card_sample_500.csv  Small sample — 500 rows (quick testing)
├── new_data.csv                    Mildly drifted data (created by new.py)
├── degraded_data.csv               Heavily degraded data (created by generate_degraded_data.py)
├── requirements.txt                Python dependencies
└── .gitignore                      Excludes .pkl and .csv from git

Generated at runtime:
├── model.pkl                       Trained XGBoost classifier
├── scaler.pkl                      Fitted StandardScaler
├── imputer.pkl                     Fitted SimpleImputer
└── baseline.pkl                    Baseline metadata (X_train + baseline_roc_auc)
```

---

## 4. Step-by-Step Pipeline

### Step A — Generate test data (`new.py`)

Creates `new_data.csv` — the original dataset with mild, realistic drift applied:

| Column | Change | Simulates |
|---|---|---|
| `LIMIT_BAL` | ×1.5 | Credit limit inflation |
| `BILL_AMT1` | ×1.3 | Higher billing amounts |
| `PAY_AMT1` | ×0.7 | Lower payment amounts |
| `AGE` | +2 years | Customer base aging |
| All numeric | + Gaussian noise (std=0.05) | Measurement variance |

Labels (`default.payment.next.month`) are **not modified** so model performance can still be evaluated.

---

### Step B — Generate degraded data (`generate_degraded_data.py`)

Creates `degraded_data.csv` — a dataset specifically designed to make the model fail. Used to test that Model Health correctly shows **"Decline"**.

Three types of degradation are applied:

| Type | What changes | Why it hurts the model |
|---|---|---|
| Heavy feature drift | LIMIT_BAL ×3, PAY delays +3 months, BILL_AMT ×2.5, PAY_AMT ×0.2, AGE +10 | Model sees distributions far outside training range |
| Concept drift | 30% of labels flipped (0→1, 1→0) | The learned feature→label relationship is now wrong |
| Heavy noise | Gaussian noise (std=0.5) on all numeric columns | Destroys the signal the model relies on |

**Result:** ROC-AUC drops from **0.85 → 0.50** (random guessing).

---

### Step C — Train XGBoost model (`train.py`)

1. Load `UCI_Credit_Card.csv`, drop `ID` column
2. Split: 70% train / 30% test (`stratify=y` keeps class ratios equal)
3. Impute missing values with mean strategy (fitted on train only)
4. Scale features with StandardScaler (fitted on train only)
5. Compute `scale_pos_weight = neg_count / pos_count` (~3.5) to handle class imbalance
6. Train XGBoost (200 trees, depth=6, learning_rate=0.05)
7. Evaluate ROC-AUC on test set — **baseline: 0.7746**
8. Save 4 artifacts: `model.pkl`, `scaler.pkl`, `imputer.pkl`, `baseline.pkl`

`baseline.pkl` stores: raw `X_train` (for drift reference) + `baseline_roc_auc`.

---

### Step D — CLI monitoring (`monitor.py`)

Run with: `python monitor.py --new_data new_data.csv`

1. Load all 4 artifacts
2. Load new CSV, separate features from labels
3. Compute PSI for every numeric feature (baseline vs new distribution)
4. Impute + scale using the **same** fitted objects (no re-fitting — that would hide drift)
5. Get probability predictions with `predict_proba`
6. Compute ROC-AUC and drop vs baseline
7. Set trend status based on PSI level
8. Trigger alert if `drift_impact_score > 0.05`

---

### Step E — Dashboard (`app.py`)

Run with: `streamlit run app.py`

Upload 5 files: `model.pkl`, `scaler.pkl`, `imputer.pkl`, `baseline.pkl`, new data CSV.

The dashboard computes drift metrics, ROC-AUC, and model health in real time and renders all charts and alerts.

---

## 5. File-by-File Deep Dive

### `train.py`

```python
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos   # ~3.5 — tells XGBoost defaults are 3.5x more costly to miss
```

`scale_pos_weight` is the key XGBoost parameter for imbalanced data. It weights the minority class (defaulters) higher during training so the model doesn't just predict "no default" for everyone.

```python
model = XGBClassifier(
    n_estimators=200,       # 200 decision trees
    max_depth=6,            # each tree can be up to 6 levels deep
    learning_rate=0.05,     # small steps = less overfitting
    scale_pos_weight=spw,   # handles class imbalance
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
```

```python
y_prob = model.predict_proba(X_test_scaled)[:, 1]   # probability of default
baseline_roc_auc = roc_auc_score(y_test, y_prob)
```

ROC-AUC is computed from probabilities, not hard predictions. This is more informative than F1 because it evaluates the model's ranking ability across all possible thresholds.

---

### `monitor.py`

**`calculate_psi(expected, actual, bins=10)`**

Bins both distributions into 10 intervals, computes what fraction of data falls in each bin, then applies the PSI formula:

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

| PSI | Interpretation |
|---|---|
| < 0.10 | No significant change |
| 0.10 – 0.25 | Moderate shift — monitor |
| > 0.25 | Major shift — investigate |

**`detect_drift_psi(baseline_df, new_df)`**

Loops every numeric column, computes PSI, returns a DataFrame sorted by PSI descending — the most drifted features appear first.

**`run_monitoring(...)`**

Key logic:
```python
# Drift impact score combines statistical drift + performance drop
drift_impact_score = (0.6 * drift_score) + (0.4 * roc_auc_drop)

# Trend status based on PSI magnitude
if drift_score > 0.25:   trend_status = "At Risk"
elif drift_score > 0.10: trend_status = "Caution"
else:                    trend_status = "Stable"

alert_triggered = drift_impact_score > 0.05
```

---

### `app.py`

**Sidebar configuration:**
- `alpha` — weight of drift vs ROC-AUC drop in the impact score (0=only performance, 1=only drift)
- `Alert Threshold` — the score above which a critical alert fires (default 0.1)
- `Rolling Window Size` — how many past batches form the dynamic baseline
- `Target Column Name` — defaults to `default.payment.next.month`
- File uploaders for all 5 artifacts

**Drift calculation:**
```python
# Feature importance from XGBoost used to weight drift scores
feature_importance = dict(zip(X_new.columns, model.feature_importances_))

# For each feature: compute unified drift score
u_score, psi, kl, js = get_unified_drift(reference_col, new_col)

# Impact Rank = drift × how much the model depends on that feature
'Impact Rank': u_score * importance
```

**Preprocessing before prediction:**
```python
if imputer and scaler:
    X_proc = pd.DataFrame(
        scaler.transform(imputer.transform(X_new)),
        columns=X_new.columns
    )
```
This is critical — without applying the same scaler used during training, predictions are meaningless.

**Model health logic:**
```python
if current_roc_auc is not None and roc_auc_drop > 0.02:
    health_label = "Decline"          # actual performance fell
elif drift_impact_score > threshold:
    health_label = "At Risk"          # drift is dangerous
elif drift_impact_score > threshold * 0.5:
    health_label = "Caution"          # drift is moderate
else:
    health_label = "Stable"
```

---

### `compare_models.py`

Trains and compares Random Forest, XGBoost, and LightGBM with all improvement techniques:

1. **RF Baseline** — original RandomForest, no tuning
2. **XGBoost** — with `scale_pos_weight`
3. **XGBoost + Feature Engineering** — 8 additional domain features
4. **XGBoost + Feature Eng + SMOTE** — synthetic minority oversampling
5. **XGBoost + Feature Eng + SMOTE + Hyperparameter Tuning** — RandomizedSearchCV
6. **All + Optimal Threshold** — find best decision threshold via Precision-Recall curve

Auto-updates `train.py` with whatever model achieves the highest F1.

---

### `generate_degraded_data.py`

Loads original data and trained artifacts. Applies three degradation types, saves `degraded_data.csv`, then immediately verifies the ROC-AUC drop by running the model on the degraded data.

---

### `new.py`

Simple drift generator. Applies realistic macro shifts (inflation, aging, payment stress) + small Gaussian noise. Labels are kept clean.

---

## 6. Dashboard — Charts & Graphs Explained

### Metric Cards (top row, 4 columns)

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Unified Drift   │ │ Drift Impact    │ │ ROC-AUC         │ │ Model Health    │
│ Score           │ │ Score           │ │                 │ │                 │
│ 0.7391          │ │ 0.4434          │ │ 0.7778          │ │ At Risk         │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Unified Drift Score**
Average of all features' unified drift scores. Higher = more overall distribution shift.
Formula: `0.5×PSI + 0.25×KL + 0.25×JS` per feature, then averaged.
- Near 0 → data looks like training data
- Above 0.5 → significant shift across the board

**Drift Impact Score**
Combines statistical drift with actual performance drop:
`α × unified_drift + (1-α) × roc_auc_drop`
Controlled by the alpha slider. If alpha=0.6: 60% drift, 40% performance.
This is the single number that triggers alerts — if it exceeds the alert threshold, a red alert fires.

**ROC-AUC**
The model's ranking ability on the new batch.
- Shows current value vs baseline (e.g., `baseline 0.7746`)
- Requires scaler.pkl + imputer.pkl to be uploaded for accurate computation
- Range: 0.5 = random, 1.0 = perfect
- A drop > 0.02 from baseline triggers "Decline" in Model Health

**Model Health**
Single status label summarizing overall model trustworthiness:
- `Stable` — everything normal
- `Caution` — drift_impact_score > threshold × 0.5
- `At Risk` — drift_impact_score > threshold
- `Decline` — ROC-AUC has actually dropped > 0.02 from baseline

---

### Alert Banner

Appears below the metric cards. Three states:

- **Red (CRITICAL ALERT)** — `drift_impact_score > threshold`. The combined drift+performance signal has crossed your configured limit. Retraining recommended.
- **Yellow (DECAY WARNING)** — Model Health is Decline or At Risk but drift_impact_score is within threshold.
- **Green (Stable)** — No significant issues detected.

---

### Feature-Level Drift Analysis Table (left column)

Shows the top 10 most impactful drifted features, sorted by **Impact Rank**.

| Column | What it means |
|---|---|
| Feature | Column name from the dataset |
| Unified Drift | Combined PSI+KL+JS drift score for this feature |
| PSI | Population Stability Index — how much the histogram shifted |
| KL | Kullback-Leibler divergence — asymmetric distance between distributions |
| JS | Jensen-Shannon divergence — symmetric, bounded 0–1 |
| Importance | How much the model relies on this feature (from `feature_importances_`) |
| Impact Rank | Unified Drift × Importance — the most actionable column |

**Color gradient on Unified Drift** — darker red = more drift. Lets you visually scan which features are most problematic.

**Why Impact Rank matters:** A feature can drift heavily but if the model ignores it, it doesn't matter. Conversely, a feature with moderate drift but very high importance is dangerous. Impact Rank = `drift × importance` captures both dimensions.

---

### ROC Curve (right column, top)

```
True Positive Rate
1.0 |        ___----------
    |    ___/
    |   /   AUC = 0.7778
    |  /
    | /
0.0 |/___________________
    0.0               1.0
         False Positive Rate
```

**What it shows:** At every possible decision threshold, the ROC curve plots how many real defaulters the model catches (True Positive Rate / Recall) vs. how many non-defaulters it incorrectly flags (False Positive Rate).

- **Diagonal line** — random guessing baseline (AUC = 0.5). A useless model.
- **Blue curve** — your model. The more it bows toward the top-left corner, the better.
- **AUC value** — area under the curve. Probability that the model ranks a random defaulter above a random non-defaulter.

**How to read it for business:**
- Moving left along the curve: stricter threshold → fewer false alarms but miss more real defaulters
- Moving right along the curve: looser threshold → catch more defaulters but more false alarms
- The optimal operating point depends on the business cost: how expensive is a missed defaulter vs a false alert?

**When it changes:** If the ROC curve on new data dips closer to the diagonal, the model is losing discriminative power — time to retrain.

---

### Drift Metric Breakdown Bar Chart (right column, bottom)

```
PSI  ████████████████ 0.74
KL   ████████████████████████ 1.12
JS   ████████ 0.38
```

Shows the **average** PSI, KL, and JS across all features. Helps understand which type of drift signal is strongest.

**PSI bar** — volume/proportion shift. If PSI is high, many features have significantly different bin distributions from baseline.

**KL bar** — this value can be large (5, 10, 20+) because KL is unbounded. A large KL means some bins have near-zero probability in one distribution but not the other — extreme distributional differences.

**JS bar** — always between 0 and ln(2) ≈ 0.693. A normalized, symmetric version of KL. Easier to compare across features. Values above 0.3 indicate strong distributional difference.

**Reading the chart together:**
- PSI high + JS high + KL high → severe, consistent drift across all features
- KL high but JS low → KL is being inflated by a few extreme bins (outliers) — JS gives the more reliable picture
- All low → data looks similar to training, drift is not the issue

---

### Rolling Baseline Button

**"Commit to Rolling Baseline"** — appends the current batch to the rolling buffer.

The reference distribution used for drift calculation is not always the original training data. If you upload batch after batch, this button lets you "accept" the current data as the new normal. The buffer keeps the last `window_size` batches and uses their combined distribution as the reference.

**Use case:** Seasonal data. If your customer base genuinely ages every year, committing quarterly batches prevents false drift alarms from expected shifts.

---

## 7. Key Concepts

### Why ROC-AUC instead of F1

| | ROC-AUC | F1 |
|---|---|---|
| Affected by class imbalance | No | Yes, heavily |
| Threshold-dependent | No | Yes (default 0.5 is arbitrary) |
| Standard in finance | Yes | Less common |
| Score on this dataset | 0.7746 (good) | 0.52 (looks bad due to imbalance) |

F1 is low because the dataset is imbalanced (22% default rate). The model predicting "no default" for everyone would get 78% accuracy. ROC-AUC is not fooled — it measures ranking ability, not binary prediction at a fixed threshold.

### PSI (Population Stability Index)

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

Compares histograms of baseline vs new data. The standard metric in credit risk monitoring.

### KL Divergence

```
KL(P || Q) = Σ P(i) × ln(P(i) / Q(i))
```

Asymmetric — KL(P||Q) ≠ KL(Q||P). Unbounded — can be very large if distributions barely overlap.

### JS Divergence

```
M = 0.5 × (P + Q)
JS = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
```

Symmetric and bounded (0 to ln(2) ≈ 0.693). More robust than KL for extreme differences.

### Unified Drift Score

```
Unified = 0.5 × PSI + 0.25 × KL + 0.25 × JS
```

Blends all three metrics. PSI gets highest weight (0.5) as it is the industry standard.

### Drift Impact Score

```
Drift Impact = α × avg_unified_drift + (1 − α) × roc_auc_drop
```

Combines statistical evidence (distribution shift) with real observed performance degradation. Configurable via the `alpha` slider.

### Model Health States

| State | Trigger |
|---|---|
| Stable | drift_impact_score ≤ threshold × 0.5 |
| Caution | drift_impact_score > threshold × 0.5 |
| At Risk | drift_impact_score > threshold |
| Decline | ROC-AUC dropped > 0.02 from baseline |

---

## 8. Datasets

### `UCI_Credit_Card.csv` — Baseline training data

30,000 rows × 24 columns (after dropping ID). Binary classification: predict credit card default.

| Column group | Columns | Description |
|---|---|---|
| Target | `default.payment.next.month` | 0 = no default, 1 = default (~22% positive) |
| Credit | `LIMIT_BAL` | Credit limit amount |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Customer attributes |
| Payment status | `PAY_0`, `PAY_2`–`PAY_6` | Repayment status past 6 months |
| Bill amounts | `BILL_AMT1`–`BILL_AMT6` | Statement balance past 6 months |
| Payment amounts | `PAY_AMT1`–`PAY_AMT6` | Amount paid past 6 months |

### `new_data.csv` — Mild drift data

Same structure. LIMIT_BAL +50%, BILL_AMT1 +30%, PAY_AMT1 −30%, AGE +2, small noise.
ROC-AUC remains near baseline (drift detected but performance holds).

### `degraded_data.csv` — Severe degradation data

Same structure. Three simultaneous degradations applied.
ROC-AUC drops from **0.85 → 0.50**. Model Health shows **"Decline"**.

---

## 9. Model Evolution — Why XGBoost

The project started with RandomForest. After systematic comparison:

| Model | ROC-AUC | F1 | Why |
|---|---|---|---|
| Random Forest | 0.757 | 0.462 | Ignores class imbalance |
| XGBoost + scale_pos_weight | **0.775** | **0.533** | Handles imbalance natively |
| LightGBM + Optuna | 0.779 | 0.540 | Marginally better but more complex |

**XGBoost was chosen** because:
- +1.8% ROC-AUC over RF (meaningful at production scale)
- +7.1% F1 over RF
- `scale_pos_weight` directly addresses class imbalance
- Simple, no extra dependencies vs LightGBM

### Why F1 is still low (0.53)

The dataset is inherently imbalanced (22% default rate) and noisy — people default due to job loss, illness, divorce — events not in the data. The features simply cannot perfectly predict default. ROC-AUC of 0.775 is the honest measure of how well the signal in the data can be extracted.

### Techniques explored in `compare_models.py`

| Technique | F1 impact | Notes |
|---|---|---|
| Feature engineering (+8 features) | +0.007 on RF | Marginal on XGBoost — it learns interactions itself |
| SMOTE | −0.02 | Hurt — synthetic samples confused the model |
| Hyperparameter tuning (RandomizedSearchCV) | +0.005 | Some gain |
| Optimal threshold tuning | +0.012 | Best threshold ~0.58, not 0.50 |
| LightGBM + Optuna (50 trials) | +0.008 vs XGB | Marginal gain, more complexity |

---

## 10. Generated Artifacts

| File | Created by | Contains | Used by |
|---|---|---|---|
| `model.pkl` | `train.py` | Trained XGBoost classifier | `monitor.py`, `app.py` |
| `scaler.pkl` | `train.py` | Fitted StandardScaler | `monitor.py`, `app.py` |
| `imputer.pkl` | `train.py` | Fitted SimpleImputer | `monitor.py`, `app.py` |
| `baseline.pkl` | `train.py` | `X_train` + `baseline_roc_auc` | `monitor.py`, `app.py` |
| `new_data.csv` | `new.py` | 30K rows, mild drift | `monitor.py`, `app.py` |
| `degraded_data.csv` | `generate_degraded_data.py` | 30K rows, severe degradation | `app.py` (for testing Decline) |

All `.pkl` and `.csv` files are excluded from git via `.gitignore`.

---

## 11. Running the Project

### Setup

```bash
pip install -r requirements.txt
pip install xgboost lightgbm optuna imbalanced-learn
```

### Full workflow

```bash
# Generate test data
python new.py                          # mild drift -> new_data.csv
python generate_degraded_data.py       # severe degradation -> degraded_data.csv

# Train model
python train.py                        # trains XGBoost, saves 4 .pkl files

# Monitor via CLI
python monitor.py --new_data new_data.csv
python monitor.py --new_data degraded_data.csv

# Compare models (optional, updates train.py with best model)
python compare_models.py

# Launch dashboard
streamlit run app.py
```

### Dashboard upload order

1. `model.pkl`
2. `scaler.pkl`
3. `imputer.pkl`
4. `baseline.pkl`
5. `new_data.csv` → expect Stable/At Risk
   OR
   `degraded_data.csv` → expect Decline

### Expected CLI output (new_data.csv)

```
INFO - Drift Score (Avg PSI) : 0.7391
INFO - ROC-AUC              : 0.7778  (baseline: 0.7746, drop: 0.0000)
INFO - Drift Impact Score   : 0.4434
INFO - Trend Status         : At Risk — high drift detected
INFO - Alert Triggered      : Yes
```

### Expected CLI output (degraded_data.csv)

```
INFO - Drift Score (Avg PSI) : ~2.0+
INFO - ROC-AUC              : ~0.50  (baseline: 0.7746, drop: ~0.27)
INFO - Drift Impact Score   : high
INFO - Trend Status         : At Risk — high drift detected
INFO - Alert Triggered      : Yes
```
