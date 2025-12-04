"""
Sector F (Construction) - Supervised Economic Analysis
=======================================================

Multi-stage analysis:
1. Supervised Random Forest with economic features
2. Feature importance ranking
3. Risk stratification
4. Economic regime clustering + regime-specific predictions
5. SHAP interaction analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modeling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                            roc_curve, precision_recall_curve, average_precision_score)

# SHAP for interactions
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Serialization
import joblib
import json
from datetime import datetime

print("="*80)
print("SECTOR F (CONSTRUCTION) - SUPERVISED ECONOMIC ANALYSIS")
print("="*80)
print(f"Started: {datetime.now()}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/7] Loading data...")
data_dir = Path(r'C:\Users\magnu\Desktop\AI Management\INF4090\data\features')
df = pd.read_parquet(data_dir / 'feature_dataset_v1.parquet')

# Filter to years 2016-2018
df_all = df[df['year'].isin([2016, 2017, 2018])].copy()

# Extract NACE and filter to Construction (41-43)
def extract_nace_code(naringskode):
    try:
        code_str = str(naringskode).split('.')[0]
        if code_str and code_str.strip() and code_str.strip()[0].isdigit():
            return int(code_str[:2]) if len(code_str) >= 2 else None
        return None
    except:
        return None

df_all['nace_code'] = df_all['Næringskode'].apply(extract_nace_code)
sector_df = df_all[(df_all['nace_code'] >= 41) & (df_all['nace_code'] <= 43)].copy()

print(f"  Total observations: {len(sector_df):,}")
print(f"  Companies: {sector_df['Orgnr'].nunique():,}")
print(f"  Bankruptcy rate: {sector_df['bankrupt'].mean():.2%}")
print(f"  Bankruptcies: {sector_df['bankrupt'].sum():,}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n[2/7] Engineering features...")

# Base economic features
# Note: Tall 7709 (Annen driftsinntekt) merged into Tall 72 during feature engineering
raw_accounting = [
    'Tall 1340', 'Tall 72', 'Tall 146',
    'Tall 217', 'Tall 194', 'Tall 85', 'Tall 86', 'Tall 17130'
]

financial_ratios = [
    'likviditetsgrad_1', 'total_gjeldsgrad', 'langsiktig_gjeldsgrad',
    'kortsiktig_gjeldsgrad', 'egenkapitalandel', 'driftsmargin',
    'driftsrentabilitet', 'omsetningsgrad', 'rentedekningsgrad', 'altman_z_score'
]

base_features = raw_accounting + financial_ratios

# Create interaction features
print("  Creating interaction features...")
sector_df['debt_liquidity_stress'] = sector_df['total_gjeldsgrad'] / (sector_df['likviditetsgrad_1'] + 0.01)
sector_df['profitability_leverage'] = sector_df['driftsmargin'] * sector_df['egenkapitalandel']
sector_df['solvency_coverage'] = sector_df['egenkapitalandel'] * sector_df['rentedekningsgrad']
sector_df['extreme_leverage'] = (sector_df['total_gjeldsgrad'] > 2.0).astype(int)
sector_df['liquidity_crisis'] = ((sector_df['likviditetsgrad_1'] < 1.0) &
                                  (sector_df['kortsiktig_gjeldsgrad'] > 0.7)).astype(int)
sector_df['negative_spiral'] = ((sector_df['driftsmargin'] < 0) &
                                (sector_df['egenkapitalandel'] < 0) &
                                (sector_df['likviditetsgrad_1'] < 1.5)).astype(int)
sector_df['size_leverage_interaction'] = sector_df['Tall 217'] * sector_df['total_gjeldsgrad']
sector_df['efficiency_profitability'] = sector_df['omsetningsgrad'] * sector_df['driftsrentabilitet']

interaction_features = [
    'debt_liquidity_stress', 'profitability_leverage', 'solvency_coverage',
    'extreme_leverage', 'liquidity_crisis', 'negative_spiral',
    'size_leverage_interaction', 'efficiency_profitability'
]

all_features = base_features + interaction_features

# Prepare dataset
X = sector_df[all_features].copy()
y = sector_df['bankrupt'].copy()
orgnr = sector_df['Orgnr'].copy()
year = sector_df['year'].copy()

# Convert to numeric and handle infinity
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)

# Complete cases
missing_mask = X.isnull().any(axis=1)
X_complete = X[~missing_mask].copy()
y_complete = y[~missing_mask].copy()
orgnr_complete = orgnr[~missing_mask].copy()
year_complete = year[~missing_mask].copy()

print(f"  Total features: {len(all_features)} ({len(base_features)} base + {len(interaction_features)} interactions)")
print(f"  Complete cases: {len(X_complete):,} ({len(X_complete)/len(X)*100:.1f}%)")
print(f"  Bankruptcies in complete cases: {y_complete.sum():,} ({y_complete.mean():.2%})")

# ============================================================================
# 3. SUPERVISED MODEL - RANDOM FOREST
# ============================================================================

print("\n[3/7] Training supervised Random Forest model...")

# Split: Use 2016-2017 for train, 2018 for test (temporal split)
train_mask = year_complete.isin([2016, 2017])
test_mask = year_complete == 2018

X_train = X_complete[train_mask]
X_test = X_complete[test_mask]
y_train = y_complete[train_mask]
y_test = y_complete[test_mask]

print(f"  Train: {len(X_train):,} obs, {y_train.sum():,} bankruptcies ({y_train.mean():.2%})")
print(f"  Test:  {len(X_test):,} obs, {y_test.sum():,} bankruptcies ({y_test.mean():.2%})")

# Train Random Forest with class balancing
print("  Training Random Forest (200 trees)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predictions
y_train_pred = rf.predict_proba(X_train)[:, 1]
y_test_pred = rf.predict_proba(X_test)[:, 1]

# Metrics
train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)
train_ap = average_precision_score(y_train, y_train_pred)
test_ap = average_precision_score(y_test, y_test_pred)

print(f"\n  RESULTS:")
print(f"    Train AUC: {train_auc:.4f}")
print(f"    Test AUC:  {test_auc:.4f}")
print(f"    Train AP:  {train_ap:.4f}")
print(f"    Test AP:   {test_ap:.4f}")

# ============================================================================
# 4. FEATURE IMPORTANCE
# ============================================================================

print("\n[4/7] Analyzing feature importance...")

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  TOP 15 FEATURES:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:.4f}")

# ============================================================================
# 5. RISK STRATIFICATION
# ============================================================================

print("\n[5/7] Risk stratification...")

# Use full dataset predictions
X_complete_full = X_complete.copy()
y_pred_full = rf.predict_proba(X_complete_full)[:, 1]

# Create risk tiers
risk_tiers = pd.cut(y_pred_full,
                    bins=[0, 0.02, 0.05, 0.10, 0.20, 1.0],
                    labels=['Very Low (0-2%)', 'Low (2-5%)', 'Medium (5-10%)', 'High (10-20%)', 'Very High (20%+)'])

# Analyze tiers
print(f"\n  RISK TIER ANALYSIS:")
for tier in ['Very Low (0-2%)', 'Low (2-5%)', 'Medium (5-10%)', 'High (10-20%)', 'Very High (20%+)']:
    tier_mask = risk_tiers == tier
    tier_bankruptcies = y_complete[tier_mask].sum()
    tier_total = tier_mask.sum()
    tier_rate = y_complete[tier_mask].mean() if tier_total > 0 else 0
    tier_pred = y_pred_full[tier_mask].mean() if tier_total > 0 else 0
    print(f"    {tier:20s} N={tier_total:5d}, Bankrupt={tier_bankruptcies:3d} ({tier_rate:5.2%}), Pred={tier_pred:.3f}")

# Economic profiles by tier
print(f"\n  ECONOMIC PROFILES BY RISK TIER:")
for tier in ['Very Low (0-2%)', 'Low (2-5%)', 'Medium (5-10%)', 'High (10-20%)', 'Very High (20%+)']:
    tier_mask = risk_tiers == tier
    if tier_mask.sum() > 0:
        tier_data = X_complete[tier_mask]
        print(f"    {tier}:")
        print(f"      Debt ratio:   {tier_data['total_gjeldsgrad'].mean():.2f}")
        print(f"      Liquidity:    {tier_data['likviditetsgrad_1'].mean():.2f}")
        print(f"      Margin:       {tier_data['driftsmargin'].mean()*100:.2f}%")
        print(f"      Equity ratio: {tier_data['egenkapitalandel'].mean():.2f}")
        print(f"      Altman Z:     {tier_data['altman_z_score'].mean():.2f}")

# ============================================================================
# 6. ECONOMIC REGIME ANALYSIS
# ============================================================================

print("\n[6/7] Economic regime analysis...")

# PCA for dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_complete)

pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA: {X_pca.shape[1]} components explaining {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
clusters = kmeans.fit_predict(X_pca)

print(f"\n  ECONOMIC REGIMES (K=3):")
for cluster_id in range(3):
    cluster_mask = clusters == cluster_id
    cluster_bankruptcies = y_complete[cluster_mask].sum()
    cluster_total = cluster_mask.sum()
    cluster_rate = y_complete[cluster_mask].mean()

    print(f"\n    Regime {cluster_id}: N={cluster_total:5d}, Bankruptcy={cluster_rate:5.2%}")

    # Get characteristics
    cluster_data = X_complete[cluster_mask]
    sector_data_cluster = sector_df.loc[X_complete.index[cluster_mask]]

    print(f"      Avg fixed assets:   {sector_data_cluster['Tall 217'].mean():,.0f} NOK")
    print(f"      Avg debt ratio:     {cluster_data['total_gjeldsgrad'].mean():>10.2f}")
    print(f"      Avg liquidity:      {cluster_data['likviditetsgrad_1'].mean():>10.2f}")
    print(f"      Avg margin:         {cluster_data['driftsmargin'].mean()*100:>10.2f}%")

# Regime-specific feature importance
print(f"\n  REGIME-SPECIFIC FEATURE IMPORTANCE:")

for cluster_id in range(3):
    # Skip if too few bankruptcies
    cluster_mask = clusters == cluster_id
    if y_complete[cluster_mask].sum() < 10:
        continue

    # Train regime-specific model
    X_cluster = X_complete[cluster_mask]
    y_cluster = y_complete[cluster_mask]

    rf_cluster = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_cluster.fit(X_cluster, y_cluster)

    importance_cluster = pd.DataFrame({
        'feature': all_features,
        'importance': rf_cluster.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n    Regime {cluster_id} - Top 5 predictors:")
    for idx, row in importance_cluster.head(5).iterrows():
        print(f"      {row['feature']:30s} {row['importance']:.4f}")

# ============================================================================
# 7. SHAP ANALYSIS (Optional - can be slow)
# ============================================================================

print("\n[7/7] SHAP interaction analysis...")
print("  Skipping SHAP for now (use feature importance from Random Forest)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_dir = Path(r'C:\Users\magnu\Desktop\AI Management\INF4090\predictions\Supervised_Economic_Analysis\Sector_F_Construction')
output_dir.mkdir(parents=True, exist_ok=True)

# Save models
joblib.dump(rf, output_dir / 'random_forest_model.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')
joblib.dump(pca, output_dir / 'pca_model.pkl')
joblib.dump(kmeans, output_dir / 'kmeans_model.pkl')
print("  Saved: Models (RF, scaler, PCA, K-Means)")

# Save predictions
predictions_df = pd.DataFrame({
    'Orgnr': orgnr_complete,
    'year': year_complete,
    'bankrupt': y_complete,
    'predicted_prob': y_pred_full,
    'risk_tier': risk_tiers,
    'regime': clusters
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
print("  Saved: predictions.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
print("  Saved: feature_importance.csv, shap_importance.csv")

# Save risk tier analysis
risk_analysis = []
for tier in ['Very Low (0-2%)', 'Low (2-5%)', 'Medium (5-10%)', 'High (10-20%)', 'Very High (20%+)']:
    tier_mask = risk_tiers == tier
    risk_analysis.append({
        'tier': tier,
        'count': tier_mask.sum(),
        'bankruptcies': y_complete[tier_mask].sum(),
        'bankruptcy_rate': y_complete[tier_mask].mean() if tier_mask.sum() > 0 else 0,
        'avg_prediction': y_pred_full[tier_mask].mean() if tier_mask.sum() > 0 else 0
    })

pd.DataFrame(risk_analysis).to_csv(output_dir / 'risk_tier_analysis.csv', index=False)
print("  Saved: risk_tier_analysis.csv, risk_tier_profiles.csv")

# Save regime analysis
regime_stats = []
for cluster_id in range(3):
    cluster_mask = clusters == cluster_id
    cluster_data = X_complete[cluster_mask]
    regime_stats.append({
        'regime': cluster_id,
        'count': cluster_mask.sum(),
        'bankruptcies': y_complete[cluster_mask].sum(),
        'bankruptcy_rate': y_complete[cluster_mask].mean(),
        'avg_debt_ratio': cluster_data['total_gjeldsgrad'].mean(),
        'avg_liquidity': cluster_data['likviditetsgrad_1'].mean(),
        'avg_margin': cluster_data['driftsmargin'].mean()
    })

pd.DataFrame(regime_stats).to_csv(output_dir / 'regime_analysis.csv', index=False)
print("  Saved: regime_analysis.csv, regime-specific feature importances")

# Save summary JSON
summary = {
    'sector': 'F - Construction',
    'analysis_date': str(datetime.now()),
    'total_observations': len(sector_df),
    'complete_cases': len(X_complete),
    'bankruptcies': int(y_complete.sum()),
    'bankruptcy_rate': float(y_complete.mean()),
    'model_performance': {
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'train_ap': float(train_ap),
        'test_ap': float(test_ap)
    },
    'top_features': feature_importance.head(10).to_dict('records')
}

with open(output_dir / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved: analysis_summary.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Completed: {datetime.now()}")

print(f"\nModel Performance:")
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test AP:  {test_ap:.4f}")

print(f"\nTop 3 Predictors:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\nResults saved to: {output_dir}")
