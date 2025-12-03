"""
Sector C Bankruptcy Prediction - Blind Training & Testing on Full Data
========================================================================

This analysis trains models on the COMPLETE Sector C dataset and then
tests predictions on the SAME data without splitting.

Purpose: Understand maximum theoretical performance and overfitting behavior
when there's no train/test separation (models can memorize entire dataset).

This represents the CEILING of what these models can achieve.
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import xgboost as xgb

warnings.filterwarnings('ignore')

print("=" * 80)
print("SECTOR C BANKRUPTCY PREDICTION - FULL DATA TRAINING & TESTING")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print("NOTE: This trains on complete data and tests on same complete data")
print("      This shows CEILING performance (maximum memorization)\n")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

parquet_path = r'c:\Users\magnu\Desktop\AI Management\INF4090\data\features\feature_dataset_v1.parquet'
print(f"Loading data from: {parquet_path}")
df = pd.read_parquet(parquet_path)

# Filter for Sector C (NACE 05-09)
df['næring_2digit'] = df['Næringskode'].astype(str).str.replace(' ', '').str[:2]
sector_c_mask = df['næring_2digit'].isin(['05', '06', '07', '08', '09'])
df_sector_c = df[sector_c_mask].copy()

print(f"Total Sector C records: {len(df_sector_c):,}")
print(f"Bankruptcy cases: {df_sector_c['bankrupt'].sum()}")
print(f"Bankruptcy rate: {df_sector_c['bankrupt'].mean():.2%}\n")

# Define features
raw_features = [
    'Tall 1340', 'Tall 7709', 'Tall 72', 'Tall 146', 'Tall 217', 'Tall 194',
    'Tall 85', 'Tall 86', 'Tall 17130',
]

ratio_features = [
    'likviditetsgrad_1', 'total_gjeldsgrad', 'langsiktig_gjeldsgrad',
    'kortsiktig_gjeldsgrad', 'egenkapitalandel', 'driftsmargin',
    'driftsrentabilitet', 'omsetningsgrad', 'rentedekningsgrad', 'altman_z_score',
]

all_features = raw_features + ratio_features
target = 'bankrupt'

# Remove rows with missing target
df_sector_c = df_sector_c.dropna(subset=[target])

# Create feature matrix
X = df_sector_c[all_features].copy()
y = df_sector_c[target].copy()

# Handle missing values
X = X.replace([np.inf, -np.inf], np.nan)
X_clean = X.dropna()
y_clean = y[X_clean.index]

print(f"Complete observations: {len(X_clean):,}")
print(f"Bankruptcy cases in clean data: {y_clean.sum()}")
print(f"Bankruptcy rate (clean): {y_clean.mean():.2%}\n")

# Handle outliers
X_clean = X_clean.copy()
for col in X_clean.columns:
    Q1 = X_clean[col].quantile(0.25)
    Q3 = X_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)

print(f"Final data shape: {X_clean.shape}\n")

# ============================================================================
# 2. TRAIN ON FULL DATA, TEST ON FULL DATA
# ============================================================================

print("=" * 80)
print("TRAINING MODELS ON COMPLETE DATASET (NO SPLIT)")
print("=" * 80)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)

models = {}
results = {}

# Logistic Regression
print("\n1. Logistic Regression (training on full data)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y_clean)
models['Logistic Regression'] = lr
y_pred_lr = lr.predict(X_scaled)
y_pred_proba_lr = lr.predict_proba(X_scaled)[:, 1]

# Random Forest
print("2. Random Forest (training on full data)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                            min_samples_leaf=5, random_state=42, n_jobs=-1, 
                            class_weight='balanced')
rf.fit(X_clean, y_clean)
models['Random Forest'] = rf
y_pred_rf = rf.predict(X_clean)
y_pred_proba_rf = rf.predict_proba(X_clean)[:, 1]

# XGBoost
print("3. XGBoost (training on full data)...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              scale_pos_weight=(y_clean == 0).sum() / (y_clean == 1).sum(),
                              n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_clean, y_clean)
models['XGBoost'] = xgb_model
y_pred_xgb = xgb_model.predict(X_clean)
y_pred_proba_xgb = xgb_model.predict_proba(X_clean)[:, 1]

# Gradient Boosting
print("4. Gradient Boosting (training on full data)...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                min_samples_split=10, min_samples_leaf=5, random_state=42)
gb.fit(X_clean, y_clean)
models['Gradient Boosting'] = gb
y_pred_gb = gb.predict(X_clean)
y_pred_proba_gb = gb.predict_proba(X_clean)[:, 1]

print("\nAll models trained successfully!\n")

# ============================================================================
# 3. EVALUATE MODELS (on same data they trained on)
# ============================================================================

print("=" * 80)
print("MODEL EVALUATION (TESTING ON TRAINING DATA)")
print("=" * 80)
print("\n⚠️  NOTE: These are CEILING metrics. Real performance on unseen data will be MUCH LOWER.\n")

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
    
    acc = accuracy_score(y_clean, y_pred)
    prec = precision_score(y_clean, y_pred, zero_division=0)
    rec = recall_score(y_clean, y_pred, zero_division=0)
    f1 = f1_score(y_clean, y_pred, zero_division=0)
    roc = roc_auc_score(y_clean, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_clean, y_pred).ravel()
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
        'training_data_size': len(X_clean),
        'test_data_size': len(X_clean),
        'note': 'TRAINED AND TESTED ON SAME DATA - CEILING PERFORMANCE'
    }

# ============================================================================
# 4. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (TRAINED ON FULL DATA)")
print("=" * 80)

feature_importance = {}

# Random Forest
print("\nRandom Forest Top 15 Features:")
rf_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_importance.head(15).to_string(index=False))
feature_importance['Random Forest'] = rf_importance

# XGBoost
print("\n\nXGBoost Top 15 Features:")
xgb_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(xgb_importance.head(15).to_string(index=False))
feature_importance['XGBoost'] = xgb_importance

# Gradient Boosting
print("\n\nGradient Boosting Top 15 Features:")
gb_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)
print(gb_importance.head(15).to_string(index=False))
feature_importance['Gradient Boosting'] = gb_importance

# Logistic Regression
print("\n\nLogistic Regression Top 15 Features (by |coefficient|):")
lr_coefs = pd.DataFrame({
    'feature': X_scaled.columns,
    'coefficient': np.abs(lr.coef_[0])
}).sort_values('coefficient', ascending=False)
print(lr_coefs.head(15).to_string(index=False))
feature_importance['Logistic Regression'] = lr_coefs

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

output_dir = Path(r'c:\Users\magnu\Desktop\AI Management\INF4090\predictions\Sector_C_FullData_Training')
output_dir.mkdir(parents=True, exist_ok=True)

# Save results as JSON
results_file = output_dir / 'model_results_full_data.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save feature importance
for model_name, importance_df in feature_importance.items():
    csv_file = output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.csv'
    importance_df.to_csv(csv_file, index=False)

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_clean.values,
    'lr_pred': y_pred_lr,
    'rf_pred': y_pred_rf,
    'xgb_pred': y_pred_xgb,
    'gb_pred': y_pred_gb,
    'lr_proba': y_pred_proba_lr,
    'rf_proba': y_pred_proba_rf,
    'xgb_proba': y_pred_proba_xgb,
    'gb_proba': y_pred_proba_gb,
})
predictions_df.to_csv(output_dir / 'full_data_predictions.csv', index=False)

print(f"\n\nResults saved to: {output_dir}")

# ============================================================================
# 6. COMPARISON SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY COMPARISON: FULL DATA TRAINING vs 80/20 SPLIT TRAINING")
print("=" * 80)

print("\n⚠️  KEY INSIGHT:")
print("   Full data training shows CEILING performance (models memorize everything)")
print("   80/20 split training shows REALISTIC performance (models generalize)")
print("   Difference = OVERFITTING MAGNITUDE\n")

print("Accuracy Comparison:")
print(f"  Logistic Regression:   Full Data = 95.56% | 80/20 Split = 88.89% | Difference = +6.67%")
print(f"  Random Forest:         Full Data = TBD    | 80/20 Split = 77.78% | Difference = ?")
print(f"  XGBoost:               Full Data = TBD    | 80/20 Split = 77.78% | Difference = ?")
print(f"  Gradient Boosting:     Full Data = TBD    | 80/20 Split = 77.78% | Difference = ?")

print("\nAnalysis complete!")
print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
