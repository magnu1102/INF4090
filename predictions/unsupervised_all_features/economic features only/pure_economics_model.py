"""
Pure Economic Fundamentals Model
=================================

FOCUS: Only economic/financial indicators - NO company characteristics, NO filing behavior

Features used:
1. Raw accounting data (Tall fields - balance sheet, income statement, cash flow)
2. Engineered financial ratios (liquidity, leverage, profitability, Altman Z-Score)

EXCLUDED:
- Missingness indicators
- Filing behavior
- Company age, size, location
- Industry codes
- Auditor information
- Temporal features (growth - these depend on filing multiple years)

Goal: Pure economic distress patterns based on financial health ONLY

Optimizations:
- Parallel processing for K-Means (n_jobs=-1)
- Reduced PCA components (20 instead of 30)
- Skip DBSCAN (too slow)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PURE ECONOMIC FUNDAMENTALS MODEL")
print("="*70)

RANDOM_STATE = 42
N_PCA_COMPONENTS = 20  # Reduced for speed
N_CLUSTERS_RANGE = range(2, 6)  # Reduced range for speed

# Load data
script_dir = Path(__file__).parent
input_file = script_dir.parent.parent / 'data' / 'features' / 'feature_dataset_v1.parquet'

print(f"\nLoading data from: {input_file}")
df = pd.read_parquet(input_file)

df_all = df[df['year'].isin([2016, 2017, 2018])].copy()
print(f"\nUsing all years (2016-2018): {len(df_all):,} observations")
print(f"Bankruptcy rate: {df_all['bankrupt'].mean():.2%}")

target = df_all['bankrupt'].copy()
identifiers = df_all[['Orgnr', 'year']].copy()

# ============================================================================
# FEATURE SELECTION: PURE ECONOMICS ONLY
# ============================================================================

print("\n" + "="*70)
print("SELECTING PURE ECONOMIC FEATURES")
print("="*70)

# 1. RAW ACCOUNTING DATA (Tall fields only, no descriptions)
raw_accounting = [col for col in df_all.columns if col.startswith('Tall ') and 'beskrivelse' not in col.lower()]

# 2. FINANCIAL RATIOS (calculated from accounting data)
financial_ratios = [
    # Liquidity
    'likviditetsgrad_1',           # Current ratio
    'likviditetsgrad_2',           # Quick ratio

    # Leverage
    'total_gjeldsgrad',            # Total debt ratio
    'langsiktig_gjeldsgrad',       # Long-term debt ratio
    'kortsiktig_gjeldsgrad',       # Short-term debt ratio
    'egenkapitalandel',            # Equity ratio

    # Profitability
    'driftsmargin',                # Operating margin
    'totalkapitalrentabilitet',    # Return on assets

    # Efficiency
    'omsetningsgrad',              # Asset turnover

    # Coverage
    'rentedekningsgrad',           # Interest coverage

    # Composite
    'altman_z_score',              # Altman Z-Score
]

# 3. WARNING SIGNALS (binary indicators of financial distress)
warning_signals = [
    'negativ_egenkapital',         # Negative equity
    'sterkt_overbelånt',           # Debt ratio > 0.8
    'lav_likviditet',              # Current ratio < 1.0
    'driftsunderskudd',            # Operating loss
]

# Combine all economic features
all_economic_features = raw_accounting + financial_ratios + warning_signals

# Keep only features that exist in dataset
economic_features = [f for f in all_economic_features if f in df_all.columns]

print(f"\nSelected features:")
print(f"  Raw accounting fields (Tall): {len([f for f in economic_features if f.startswith('Tall')])}")
print(f"  Financial ratios: {len([f for f in financial_ratios if f in df_all.columns])}")
print(f"  Warning signals: {len([f for f in warning_signals if f in df_all.columns])}")
print(f"  TOTAL: {len(economic_features)}")

print(f"\nEXCLUDED:")
print(f"  - Missingness indicators")
print(f"  - Filing behavior features")
print(f"  - Company characteristics (age, size)")
print(f"  - Industry/location codes")
print(f"  - Temporal/growth features")
print(f"  - Auditor information")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "="*70)
print("PREPARING DATA")
print("="*70)

X = df_all[economic_features].copy()

# Convert to numeric (in case any are stored as objects)
print("\nConverting to numeric types...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle infinity and extreme values
print("Handling infinity and extreme values...")
for col in X.columns:
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    if X[col].notna().any():
        upper_cap = X[col].quantile(0.999)
        lower_cap = X[col].quantile(0.001)
        X[col] = X[col].clip(lower=lower_cap, upper=upper_cap)

numeric_missing = X.isnull().sum().sum()
total_values = X.shape[0] * X.shape[1]
print(f"Missing values: {numeric_missing:,} ({numeric_missing/total_values*100:.2f}%)")

# Impute with median
print("\nImputing missing values with median...")
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"\nFinal feature matrix: {X_imputed.shape}")
print(f"  Features: {X_imputed.shape[1]}")
print(f"  Observations: {X_imputed.shape[0]:,}")
print(f"  Missing values remaining: {X_imputed.isnull().sum().sum()}")

# Standardize
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

feature_names = X_imputed.columns.tolist()

# ============================================================================
# PCA
# ============================================================================

print("\n" + "="*70)
print("PCA DIMENSIONALITY REDUCTION")
print("="*70)

print(f"\nReducing from {X_scaled.shape[1]} to {N_PCA_COMPONENTS} dimensions...")
pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nPCA Results:")
print(f"  PC1 explains: {explained_var[0]*100:.2f}% of variance")
print(f"  First 5 PCs: {cumulative_var[4]*100:.2f}% of variance")
print(f"  All {N_PCA_COMPONENTS} PCs: {cumulative_var[-1]*100:.2f}% of variance")

# ============================================================================
# K-MEANS CLUSTERING (with parallel processing)
# ============================================================================

print("\n" + "="*70)
print("K-MEANS CLUSTERING (PARALLEL)")
print("="*70)

best_k = 2
best_silhouette = -1

print("\nTrying different numbers of clusters...")
for k in N_CLUSTERS_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    silhouette = silhouette_score(X_pca, labels)
    print(f"  k={k}: Silhouette = {silhouette:.4f}")

    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

print(f"\nBest k={best_k} (Silhouette: {best_silhouette:.4f})")

# Final clustering
print(f"\nFitting final K-Means with k={best_k}...")
kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_pca)

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CLUSTER ANALYSIS - ECONOMIC PATTERNS")
print("="*70)

cluster_df = pd.DataFrame({
    'cluster': kmeans_labels,
    'bankrupt': target
})

cluster_stats = cluster_df.groupby('cluster').agg({
    'bankrupt': ['count', 'sum', 'mean']
}).round(4)
cluster_stats.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
print("\nCluster sizes and bankruptcy rates:")
print(cluster_stats)

# ============================================================================
# PCA INTERPRETATION
# ============================================================================

print("\n" + "="*70)
print("PCA INTERPRETATION - WHAT DRIVES BANKRUPTCY?")
print("="*70)

for i in range(min(3, N_PCA_COMPONENTS)):
    loadings = pd.DataFrame({
        'Feature': feature_names,
        'Loading': pca.components_[i]
    }).sort_values('Loading', key=abs, ascending=False)

    print(f"\n{'='*70}")
    print(f"PC{i+1} (explains {explained_var[i]*100:.2f}% variance)")
    print(f"{'='*70}")

    print("\nTop 10 features contributing to this dimension:")
    for idx, row in loadings.head(10).iterrows():
        direction = "+" if row['Loading'] > 0 else "-"
        print(f"  {direction} {row['Feature']:40s} {abs(row['Loading']):.4f}")

# ============================================================================
# CLUSTER PROFILING
# ============================================================================

print("\n" + "="*70)
print("CLUSTER PROFILING - FINANCIAL CHARACTERISTICS")
print("="*70)

df_profiling = X_imputed.copy()
df_profiling['cluster'] = kmeans_labels
df_profiling['bankrupt'] = target.values

# Profile key financial ratios
key_ratios = [
    'altman_z_score',
    'likviditetsgrad_1',
    'total_gjeldsgrad',
    'egenkapitalandel',
    'driftsmargin',
    'totalkapitalrentabilitet',
    'negativ_egenkapital',
]

available_ratios = [r for r in key_ratios if r in df_profiling.columns]

print("\nMean financial ratios by cluster:")
print("-" * 70)

for ratio in available_ratios:
    print(f"\n{ratio}:")
    cluster_means = df_profiling.groupby('cluster')[ratio].agg(['mean', 'median', 'std']).round(4)
    for cluster_id in cluster_means.index:
        bankruptcy_rate = cluster_stats.loc[cluster_id, 'Bankruptcy_Rate']
        mean_val = cluster_means.loc[cluster_id, 'mean']
        median_val = cluster_means.loc[cluster_id, 'median']
        print(f"  Cluster {cluster_id} (Bankr: {bankruptcy_rate:6.1%}): Mean={mean_val:8.4f}, Median={median_val:8.4f}")

# Identify most distinctive features
print("\n" + "="*70)
print("MOST DISTINCTIVE FEATURES BETWEEN CLUSTERS")
print("="*70)

# Calculate variance in means across clusters
cluster_means_all = df_profiling.groupby('cluster')[feature_names].mean()
variance_across_clusters = cluster_means_all.var(axis=0).sort_values(ascending=False)

print("\nTop 20 features with highest variance across clusters:")
print("(These features best distinguish healthy from distressed companies)\n")

for idx, (feature, var) in enumerate(variance_across_clusters.head(20).items(), 1):
    # Show mean for each cluster
    cluster_vals = cluster_means_all[feature]
    vals_str = ", ".join([f"C{i}={v:.2f}" for i, v in cluster_vals.items()])
    print(f"{idx:2d}. {feature:40s} [{vals_str}]")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = script_dir

# Cluster assignments
results_df = pd.DataFrame({
    'Orgnr': identifiers['Orgnr'],
    'year': identifiers['year'],
    'bankrupt': target,
    'cluster': kmeans_labels
})
results_df.to_csv(output_dir / 'pure_economic_cluster_assignments.csv', index=False)
print(f"Cluster assignments saved to: pure_economic_cluster_assignments.csv")

# PCA components
pca_df = pd.DataFrame(
    X_pca[:, :10],
    columns=[f'PC{i+1}' for i in range(min(10, N_PCA_COMPONENTS))]
)
pca_df['Orgnr'] = identifiers['Orgnr'].values
pca_df['year'] = identifiers['year'].values
pca_df['bankrupt'] = target.values
pca_df['cluster'] = kmeans_labels
pca_df.to_csv(output_dir / 'pure_economic_pca_components.csv', index=False)
print(f"PCA components saved to: pure_economic_pca_components.csv")

# PCA loadings
all_loadings = []
for i in range(N_PCA_COMPONENTS):
    loadings = pd.DataFrame({
        'Component': f'PC{i+1}',
        'Feature': feature_names,
        'Loading': pca.components_[i],
        'Abs_Loading': np.abs(pca.components_[i])
    }).sort_values('Abs_Loading', ascending=False)
    all_loadings.append(loadings)

pd.concat(all_loadings).to_csv(output_dir / 'pure_economic_pca_loadings.csv', index=False)
print(f"PCA loadings saved to: pure_economic_pca_loadings.csv")

# Cluster profiles
cluster_profiles = df_profiling.groupby('cluster')[available_ratios].agg(['mean', 'median', 'std']).round(4)
cluster_profiles.to_csv(output_dir / 'pure_economic_cluster_profiles.csv')
print(f"Cluster profiles saved to: pure_economic_cluster_profiles.csv")

# Summary
summary = {
    'model': 'Pure Economic Fundamentals Model',
    'date': datetime.now().isoformat(),
    'focus': 'Pure economic/financial distress (NO company characteristics, NO filing behavior)',
    'data': {
        'total_observations': len(df_all),
        'bankruptcy_rate': float(target.mean())
    },
    'features': {
        'raw_accounting_fields': len([f for f in economic_features if f.startswith('Tall')]),
        'financial_ratios': len([f for f in financial_ratios if f in df_all.columns]),
        'warning_signals': len([f for f in warning_signals if f in df_all.columns]),
        'total': len(economic_features),
        'excluded_types': [
            'missingness_indicators',
            'filing_behavior',
            'company_characteristics',
            'temporal_growth',
            'industry_location',
            'auditor_info'
        ]
    },
    'pca': {
        'n_components': N_PCA_COMPONENTS,
        'variance_explained': float(cumulative_var[-1]),
        'pc1_variance': float(explained_var[0])
    },
    'clustering': {
        'algorithm': 'K-Means (parallel processing)',
        'best_k': int(best_k),
        'silhouette_score': float(best_silhouette),
        'cluster_sizes': cluster_stats['Companies'].tolist(),
        'cluster_bankruptcy_rates': cluster_stats['Bankruptcy_Rate'].tolist()
    },
    'most_distinctive_features': variance_across_clusters.head(10).to_dict()
}

with open(output_dir / 'pure_economic_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: pure_economic_results.json")

print("\n" + "="*70)
print("PURE ECONOMIC MODEL COMPLETE")
print("="*70)
print(f"\nKey Achievement:")
print(f"  Discovered bankruptcy patterns based ONLY on financial health")
print(f"  NO filing behavior, NO company size, NO industry effects")
print(f"  {best_k} clusters found with silhouette score {best_silhouette:.4f}")
print(f"  Used {len(economic_features)} pure economic features")
print(f"  Processed {len(df_all):,} observations")

# Final insight
print(f"\nBankruptcy rates by cluster:")
for cluster_id in range(best_k):
    count = cluster_stats.loc[cluster_id, 'Companies']
    rate = cluster_stats.loc[cluster_id, 'Bankruptcy_Rate']
    print(f"  Cluster {cluster_id}: {rate*100:5.2f}% bankruptcy ({int(count):,} companies)")
