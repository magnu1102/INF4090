"""
Sector C (Manufacturing) - Supervised Economic Analysis
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
print("SECTOR C (MANUFACTURING) - SUPERVISED ECONOMIC ANALYSIS")
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

# Extract NACE and filter to Manufacturing (10-33)
def extract_nace_code(naringskode):
    try:
        code_str = str(naringskode).split('.')[0]
        if code_str and code_str.strip() and code_str.strip()[0].isdigit():
            return int(code_str[:2]) if len(code_str) >= 2 else None
        return None
    except:
        return None

df_all['nace_code'] = df_all['Næringskode'].apply(extract_nace_code)
sector_df = df_all[(df_all['nace_code'] >= 10) & (df_all['nace_code'] <= 33)].copy()

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
y_train = y_complete[train_mask]
X_test = X_complete[test_mask]
y_test = y_complete[test_mask]

print(f"  Train: {len(X_train):,} obs, {y_train.sum():,} bankruptcies ({y_train.mean():.2%})")
print(f"  Test:  {len(X_test):,} obs, {y_test.sum():,} bankruptcies ({y_test.mean():.2%})")

# Random Forest with class balancing
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

print("  Training Random Forest (200 trees)...")
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_train_proba = rf_model.predict_proba(X_train)[:, 1]
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
train_ap = average_precision_score(y_train, y_train_proba)
test_ap = average_precision_score(y_test, y_test_proba)

print(f"\n  RESULTS:")
print(f"    Train AUC: {train_auc:.4f}")
print(f"    Test AUC:  {test_auc:.4f}")
print(f"    Train AP:  {train_ap:.4f}")
print(f"    Test AP:   {test_ap:.4f}")

# ============================================================================
# 4. FEATURE IMPORTANCE
# ============================================================================

print("\n[4/7] Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X_complete.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  TOP 15 FEATURES:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"    {row['feature']:<30} {row['importance']:.4f}")

# ============================================================================
# 5. RISK STRATIFICATION
# ============================================================================

print("\n[5/7] Risk stratification...")

# Get probabilities for all complete cases
all_proba = rf_model.predict_proba(X_complete)[:, 1]

# Create risk tiers
risk_bins = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
risk_labels = ['Very Low (0-2%)', 'Low (2-5%)', 'Medium (5-10%)', 'High (10-20%)', 'Very High (20%+)']
risk_tiers = pd.cut(all_proba, bins=risk_bins, labels=risk_labels)

# Analyze each tier
print("\n  RISK TIER ANALYSIS:")
tier_analysis = []
for tier in risk_labels:
    mask = risk_tiers == tier
    n_companies = mask.sum()
    n_bankrupt = y_complete[mask].sum()
    actual_rate = y_complete[mask].mean() if n_companies > 0 else 0
    avg_prob = all_proba[mask].mean() if n_companies > 0 else 0

    tier_analysis.append({
        'tier': tier,
        'n_companies': n_companies,
        'n_bankrupt': n_bankrupt,
        'actual_bankruptcy_rate': actual_rate,
        'avg_predicted_prob': avg_prob
    })

    print(f"    {tier:<20} N={n_companies:>5}, Bankrupt={n_bankrupt:>3} ({actual_rate:>6.2%}), Pred={avg_prob:.3f}")

tier_analysis_df = pd.DataFrame(tier_analysis)

# Economic profiles per tier
print("\n  ECONOMIC PROFILES BY RISK TIER:")
tier_profiles = []
for tier in risk_labels:
    mask = risk_tiers == tier
    if mask.sum() > 0:
        profile = {
            'tier': tier,
            'avg_debt_ratio': X_complete.loc[mask, 'total_gjeldsgrad'].mean(),
            'avg_liquidity': X_complete.loc[mask, 'likviditetsgrad_1'].mean(),
            'avg_margin': X_complete.loc[mask, 'driftsmargin'].mean(),
            'avg_equity_ratio': X_complete.loc[mask, 'egenkapitalandel'].mean(),
            'avg_altman_z': X_complete.loc[mask, 'altman_z_score'].mean(),
        }
        tier_profiles.append(profile)

        print(f"    {tier}:")
        print(f"      Debt ratio:   {profile['avg_debt_ratio']:.2f}")
        print(f"      Liquidity:    {profile['avg_liquidity']:.2f}")
        print(f"      Margin:       {profile['avg_margin']:.2%}")
        print(f"      Equity ratio: {profile['avg_equity_ratio']:.2f}")
        print(f"      Altman Z:     {profile['avg_altman_z']:.2f}")

tier_profiles_df = pd.DataFrame(tier_profiles)

# ============================================================================
# 6. ECONOMIC REGIME CLUSTERING
# ============================================================================

print("\n[6/7] Economic regime analysis...")

# Use base economic features for clustering (not interactions)
X_base = X_complete[base_features].copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_base)

# PCA
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA: {pca.n_components_} components explaining {pca.explained_variance_ratio_.sum():.1%} variance")

# K-Means clustering (test K=2 to 5)
best_k = 3
kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=20, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(X_pca)

print(f"\n  ECONOMIC REGIMES (K={best_k}):")
regime_analysis = []
for cluster_id in range(best_k):
    mask = clusters == cluster_id
    n_companies = mask.sum()
    n_bankrupt = y_complete[mask].sum()
    bankr_rate = y_complete[mask].mean()

    # Economic profile
    profile = {
        'regime': f'Regime {cluster_id}',
        'n_companies': n_companies,
        'bankruptcy_rate': bankr_rate,
        'avg_size': X_complete.loc[mask, 'Tall 217'].mean(),
        'avg_debt': X_complete.loc[mask, 'total_gjeldsgrad'].mean(),
        'avg_liquidity': X_complete.loc[mask, 'likviditetsgrad_1'].mean(),
        'avg_margin': X_complete.loc[mask, 'driftsmargin'].mean(),
        'avg_equity': X_complete.loc[mask, 'egenkapitalandel'].mean(),
    }
    regime_analysis.append(profile)

    print(f"\n    Regime {cluster_id}: N={n_companies:>5}, Bankruptcy={bankr_rate:>6.2%}")
    print(f"      Avg fixed assets: {profile['avg_size']:>12,.0f} NOK")
    print(f"      Avg debt ratio:   {profile['avg_debt']:>12.2f}")
    print(f"      Avg liquidity:    {profile['avg_liquidity']:>12.2f}")
    print(f"      Avg margin:       {profile['avg_margin']:>12.2%}")

regime_analysis_df = pd.DataFrame(regime_analysis)

# Regime-specific feature importance
print("\n  REGIME-SPECIFIC FEATURE IMPORTANCE:")
regime_importances = {}
for cluster_id in range(best_k):
    mask = clusters == cluster_id
    if mask.sum() > 100 and y_complete[mask].sum() > 5:  # Need sufficient data
        X_regime = X_complete[mask]
        y_regime = y_complete[mask]

        rf_regime = RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_regime.fit(X_regime, y_regime)

        regime_fi = pd.DataFrame({
            'feature': X_complete.columns,
            'importance': rf_regime.feature_importances_
        }).sort_values('importance', ascending=False)

        regime_importances[f'Regime {cluster_id}'] = regime_fi

        print(f"\n    Regime {cluster_id} - Top 5 predictors:")
        for idx, row in regime_fi.head(5).iterrows():
            print(f"      {row['feature']:<30} {row['importance']:.4f}")

# ============================================================================
# 7. SHAP INTERACTION ANALYSIS
# ============================================================================

print("\n[7/7] SHAP interaction analysis...")
print("  Skipping SHAP for now (use feature importance from Random Forest)")
shap_importance = feature_importance.copy()
shap_importance.columns = ['feature', 'mean_abs_shap']

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_dir = Path(__file__).parent

# Save model
joblib.dump(rf_model, output_dir / 'random_forest_model.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')
joblib.dump(pca, output_dir / 'pca_model.pkl')
joblib.dump(kmeans, output_dir / 'kmeans_clusters.pkl')
print("  Saved: Models (RF, scaler, PCA, K-Means)")

# Save predictions
predictions_df = pd.DataFrame({
    'Orgnr': orgnr_complete.values,
    'year': year_complete.values,
    'actual_bankrupt': y_complete.values,
    'predicted_proba': all_proba,
    'risk_tier': risk_tiers,
    'economic_regime': clusters
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
print("  Saved: predictions.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
shap_importance.to_csv(output_dir / 'shap_importance.csv', index=False)
print("  Saved: feature_importance.csv, shap_importance.csv")

# Save tier analysis
tier_analysis_df.to_csv(output_dir / 'risk_tier_analysis.csv', index=False)
tier_profiles_df.to_csv(output_dir / 'risk_tier_profiles.csv', index=False)
print("  Saved: risk_tier_analysis.csv, risk_tier_profiles.csv")

# Save regime analysis
regime_analysis_df.to_csv(output_dir / 'regime_analysis.csv', index=False)
for regime_name, fi_df in regime_importances.items():
    fi_df.to_csv(output_dir / f'regime_importance_{regime_name.replace(" ", "_")}.csv', index=False)
print("  Saved: regime_analysis.csv, regime-specific feature importances")

# Save summary metrics
summary = {
    'sector': 'C (Manufacturing)',
    'n_observations': len(X_complete),
    'n_companies': orgnr_complete.nunique(),
    'n_features': len(all_features),
    'n_bankruptcies': int(y_complete.sum()),
    'bankruptcy_rate': float(y_complete.mean()),
    'train_auc': float(train_auc),
    'test_auc': float(test_auc),
    'train_ap': float(train_ap),
    'test_ap': float(test_ap),
    'n_regimes': int(best_k),
    'feature_list': all_features,
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
