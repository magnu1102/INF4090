"""
Sector C Bankruptcy Prediction Model - Advanced Analysis
=====================================================

Predicts bankruptcy for Norwegian companies in Sector C (Mining and Quarrying, NACE 05-09)
using financial ratios and raw accounting data.

Models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("=" * 80)
print("SECTOR C BANKRUPTCY PREDICTION MODEL")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load data
parquet_path = r'c:\Users\magnu\Desktop\AI Management\INF4090\data\features\feature_dataset_v1.parquet'
print(f"Loading data from: {parquet_path}")
df = pd.read_parquet(parquet_path)

# Filter for Sector C (NACE 05-09)
df['næring_2digit'] = df['Næringskode'].astype(str).str.replace(' ', '').str[:2]
sector_c_mask = df['næring_2digit'].isin(['05', '06', '07', '08', '09'])
df_sector_c = df[sector_c_mask].copy()

print(f"Total records: {len(df):,}")
print(f"Sector C records: {len(df_sector_c):,}")
print(f"Sector C bankruptcy rate: {df_sector_c['bankrupt'].mean():.2%}")
print(f"Sector C bankrupts: {df_sector_c['bankrupt'].sum()}\n")

# Define features - Raw accounting numbers
raw_features = [
    'Tall 1340',    # Sales revenue
    'Tall 7709',    # Other operating income
    'Tall 72',      # Total income
    'Tall 146',     # Operating result
    'Tall 217',     # Fixed assets
    'Tall 194',     # Current assets
    'Tall 85',      # Short-term debt
    'Tall 86',      # Long-term debt
    'Tall 17130',   # Financial expenses
]

# Financial ratios
ratio_features = [
    'likviditetsgrad_1',      # Current ratio
    'total_gjeldsgrad',        # Total debt ratio
    'langsiktig_gjeldsgrad',   # Long-term debt ratio
    'kortsiktig_gjeldsgrad',   # Short-term debt ratio
    'egenkapitalandel',        # Equity ratio
    'driftsmargin',            # Operating margin
    'driftsrentabilitet',      # Operating ROA
    'omsetningsgrad',          # Asset turnover
    'rentedekningsgrad',       # Interest coverage
    'altman_z_score',          # Altman Z-Score
]

all_features = raw_features + ratio_features
target = 'bankrupt'

# Remove rows with missing target
df_sector_c = df_sector_c.dropna(subset=[target])
print(f"Records after removing missing target: {len(df_sector_c):,}\n")

# Create feature matrix
X = df_sector_c[all_features].copy()
y = df_sector_c[target].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

# Handle missing values - strategy: forward fill by company/year, then drop
print("Handling missing values...")
missing_before = X.isna().sum().sum()

# Fill extreme/inf values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Drop rows with ANY missing values (conservative approach for this sector)
X_clean = X.dropna()
y_clean = y[X_clean.index]

missing_after = X_clean.isna().sum().sum()
print(f"Rows with complete data: {len(X_clean):,}")
print(f"Removed {len(X) - len(X_clean):,} rows with missing values")
print(f"Final bankruptcy rate: {y_clean.mean():.2%}\n")

# ============================================================================
# 2. HANDLE OUTLIERS AND TRANSFORM
# ============================================================================

print("Handling outliers and transformations...")
X_clean = X_clean.copy()

# For each feature, clip extreme outliers (>3 std from median)
for col in X_clean.columns:
    Q1 = X_clean[col].quantile(0.25)
    Q3 = X_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)

print(f"Data shape after outlier handling: {X_clean.shape}\n")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

print("Creating train-test split (80/20 random)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

print(f"Train set: {len(X_train):,} ({y_train.mean():.2%} bankruptcy)")
print(f"Test set: {len(X_test):,} ({y_test.mean():.2%} bankruptcy)\n")

# ============================================================================
# 4. SCALE FEATURES
# ============================================================================

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================

print("=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {}
results = {}

# Logistic Regression
print("\n1. Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Random Forest
print("2. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                            min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)  # RF doesn't need scaling
models['Random Forest'] = rf
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# XGBoost
print("3. XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                              n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Gradient Boosting
print("4. Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                min_samples_split=10, min_samples_leaf=5, random_state=42)
gb.fit(X_train, y_train)
models['Gradient Boosting'] = gb
y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

print("\nAll models trained successfully!\n")

# ============================================================================
# 6. EVALUATE MODELS
# ============================================================================

print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

predictions = {
    'Logistic Regression': (y_pred_lr, y_pred_proba_lr),
    'Random Forest': (y_pred_rf, y_pred_proba_rf),
    'XGBoost': (y_pred_xgb, y_pred_proba_xgb),
    'Gradient Boosting': (y_pred_gb, y_pred_proba_gb),
}

results = {}

for model_name, (y_pred, y_pred_proba) in predictions.items():
    print(f"\n{model_name}")
    print("-" * 40)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"ROC-AUC:      {roc:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    results[model_name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = {}

# Random Forest
print("\nRandom Forest Top 15 Features:")
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_importance.head(15).to_string(index=False))
feature_importance['Random Forest'] = rf_importance

# XGBoost
print("\n\nXGBoost Top 15 Features:")
xgb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(xgb_importance.head(15).to_string(index=False))
feature_importance['XGBoost'] = xgb_importance

# Gradient Boosting
print("\n\nGradient Boosting Top 15 Features:")
gb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)
print(gb_importance.head(15).to_string(index=False))
feature_importance['Gradient Boosting'] = gb_importance

# Logistic Regression Coefficients
print("\n\nLogistic Regression Top 15 Features (by |coefficient|):")
lr_coefs = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'coefficient': np.abs(lr.coef_[0])
}).sort_values('coefficient', ascending=False)
print(lr_coefs.head(15).to_string(index=False))
feature_importance['Logistic Regression'] = lr_coefs

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

output_dir = Path(r'c:\Users\magnu\Desktop\AI Management\INF4090\predictions\Sector_C_Advanced_Models')
output_dir.mkdir(parents=True, exist_ok=True)

# Save results as JSON
results_file = output_dir / 'model_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save feature importance
for model_name, importance_df in feature_importance.items():
    csv_file = output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.csv'
    importance_df.to_csv(csv_file, index=False)

print(f"\n\nResults saved to: {output_dir}")

# Save best model predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'lr_pred': y_pred_lr,
    'rf_pred': y_pred_rf,
    'xgb_pred': y_pred_xgb,
    'gb_pred': y_pred_gb,
    'lr_proba': y_pred_proba_lr,
    'rf_proba': y_pred_proba_rf,
    'xgb_proba': y_pred_proba_xgb,
    'gb_proba': y_pred_proba_gb,
})
predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)

print("\nAnalysis complete!")
print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
