"""
Economic Fundamentals Unsupervised Learning Model
==================================================

Focus: Discover bankruptcy patterns based on ECONOMIC DISTRESS, not filing behavior

This model uses:
1. All raw accounting data (Tall fields - balance sheet, income statement)
2. Engineered financial ratios (liquidity, leverage, profitability)
3. Company characteristics (size, age, industry)

EXCLUDES:
- Missingness indicators (levert_alle_år, regnskapskomplett, etc.)
- Filing behavior features
- Temporal missingness features

Goal: Find clusters based on actual financial health, not data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ECONOMIC FUNDAMENTALS UNSUPERVISED MODEL")
print("="*70)

RANDOM_STATE = 42
N_PCA_COMPONENTS = 30
N_CLUSTERS_RANGE = range(2, 8)

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
# FEATURE SELECTION: ECONOMIC FUNDAMENTALS ONLY
# ============================================================================

print("\n" + "="*70)
print("SELECTING ECONOMIC FUNDAMENTAL FEATURES")
print("="*70)

# 1. RAW ACCOUNTING DATA (All Tall fields)
raw_accounting = [col for col in df_all.columns if col.startswith('Tall ') and 'beskrivelse' not in col.lower()]

# 2. ENGINEERED FINANCIAL RATIOS
financial_ratios = [
    'likviditetsgrad_1',
    'likviditetsgrad_2',
    'total_gjeldsgrad',
    'langsiktig_gjeldsgrad',
    'kortsiktig_gjeldsgrad',
    'egenkapitalandel',
    'driftsmargin',
    'totalkapitalrentabilitet',
    'omsetningsgrad',
    'rentedekningsgrad',
    'altman_z_score',
]

# 3. TEMPORAL/GROWTH FEATURES (economic trends, not missingness)
temporal_economic = [
    'omsetningsvekst_1617',
    'omsetningsvekst_1718',
    'aktivavekst_1617',
    'aktivavekst_1718',
    'gjeldsvekst_1617',
    'gjeldsvekst_1718',
    'omsetningsvolatilitet',
]

# 4. WARNING SIGNALS (actual economic distress, not filing behavior)
warning_signals = [
    'negativ_egenkapital',
    'sterkt_overbelånt',
    'lav_likviditet',
    'driftsunderskudd',
    'fallende_likviditet',
    'konsistent_underskudd',
    'økende_gjeldsgrad',
]

# 5. COMPANY CHARACTERISTICS
company_chars = [
    'selskapsalder',
    'nytt_selskap',
    'log_totalkapital',
    'log_omsetning',
    'Antall ansatte',
]

# 6. CATEGORICAL (industry, location, org form)
categorical_features = [
    'Næringskode',
    'Organisasjonsform',
    'Sektorkode',
    'Fylkenr',
]

# EXPLICITLY EXCLUDE MISSINGNESS INDICATORS
exclude_missingness = [
    'levert_alle_år',
    'levert_2018',
    'antall_år_levert',
    'regnskapskomplett',
    'kan_ikke_beregne_likviditet',
    'kan_ikke_beregne_gjeldsgrad',
    # Any feature ending in "_missing"
]

# Combine economic features
numeric_features = (
    raw_accounting +
    financial_ratios +
    temporal_economic +
    warning_signals +
    company_chars
)

# Remove any missingness-related features
numeric_features = [f for f in numeric_features if f in df_all.columns and f not in exclude_missingness and not f.endswith('_missing')]

# Keep only available categorical features
categorical_features = [f for f in categorical_features if f in df_all.columns]

print(f"\nSelected features:")
print(f"  Raw accounting fields: {len(raw_accounting)}")
print(f"  Financial ratios: {len([f for f in financial_ratios if f in df_all.columns])}")
print(f"  Temporal/growth: {len([f for f in temporal_economic if f in df_all.columns])}")
print(f"  Warning signals: {len([f for f in warning_signals if f in df_all.columns])}")
print(f"  Company characteristics: {len([f for f in company_chars if f in df_all.columns])}")
print(f"  Categorical: {len(categorical_features)}")
print(f"  TOTAL NUMERIC: {len(numeric_features)}")
print(f"  TOTAL CATEGORICAL: {len(categorical_features)}")

print(f"\nEXCLUDED missingness indicators and filing behavior features")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "="*70)
print("PREPARING DATA")
print("="*70)

X_numeric = df_all[numeric_features].copy()
X_categorical = df_all[categorical_features].copy()

# Convert all numeric columns to numeric type (some may be stored as object)
print("\nConverting to numeric types...")
for col in X_numeric.columns:
    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')

# Handle infinity and extreme values
print("Handling infinity and extreme values...")
for col in X_numeric.columns:
    X_numeric[col] = X_numeric[col].replace([np.inf, -np.inf], np.nan)
    if X_numeric[col].notna().any():
        upper_cap = X_numeric[col].quantile(0.999)
        lower_cap = X_numeric[col].quantile(0.001)
        X_numeric[col] = X_numeric[col].clip(lower=lower_cap, upper=upper_cap)

numeric_missing = X_numeric.isnull().sum().sum()
print(f"Missing values in numeric features: {numeric_missing:,}")

# Impute with median
print("\nImputing numeric features with median...")
numeric_imputer = SimpleImputer(strategy='median')
X_numeric_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)

# Encode categoricals
print("Encoding categorical features...")
X_categorical_encoded = pd.DataFrame(index=X_categorical.index)

for col in categorical_features:
    if col in X_categorical.columns:
        # Fill missing with 'Unknown'
        X_categorical[col] = X_categorical[col].fillna('Unknown')
        le = LabelEncoder()
        try:
            X_categorical_encoded[col] = le.fit_transform(X_categorical[col].astype(str))
            print(f"  Encoded {col}: {len(le.classes_)} unique values")
        except Exception as e:
            print(f"  Skipped {col}: {e}")

# Combine
X_combined = pd.concat([X_numeric_imputed, X_categorical_encoded], axis=1)
print(f"\nFinal feature matrix: {X_combined.shape}")
print(f"  Numeric features: {len(X_numeric_imputed.columns)}")
print(f"  Categorical features: {len(X_categorical_encoded.columns)}")
print(f"  Total: {X_combined.shape[1]}")

# Standardize
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

feature_names = X_combined.columns.tolist()

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
print(f"  First 10 PCs: {cumulative_var[min(9, len(cumulative_var)-1)]*100:.2f}% of variance")
print(f"  All {N_PCA_COMPONENTS} PCs: {cumulative_var[-1]*100:.2f}% of variance")

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

print("\n" + "="*70)
print("K-MEANS CLUSTERING")
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

kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_pca)

# Analyze clusters
cluster_df = pd.DataFrame({
    'cluster': kmeans_labels,
    'bankrupt': target
})

print("\n" + "="*70)
print("CLUSTER ANALYSIS")
print("="*70)

cluster_stats = cluster_df.groupby('cluster').agg({
    'bankrupt': ['count', 'sum', 'mean']
}).round(4)
cluster_stats.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
print(cluster_stats)

# ============================================================================
# PCA INTERPRETATION - WHAT DRIVES EACH COMPONENT?
# ============================================================================

print("\n" + "="*70)
print("PCA INTERPRETATION - TOP LOADINGS")
print("="*70)

for i in range(min(5, N_PCA_COMPONENTS)):
    loadings = pd.DataFrame({
        'Feature': feature_names,
        'Loading': pca.components_[i]
    }).sort_values('Loading', key=abs, ascending=False)

    print(f"\n{'='*70}")
    print(f"PC{i+1} (explains {explained_var[i]*100:.2f}% variance)")
    print(f"{'='*70}")
    print("\nTop 10 positive contributors:")
    for idx, row in loadings.head(10).iterrows():
        print(f"  {row['Feature']:40s} {row['Loading']:+.4f}")

    print("\nTop 10 negative contributors:")
    for idx, row in loadings.tail(10).iterrows():
        print(f"  {row['Feature']:40s} {row['Loading']:+.4f}")

# ============================================================================
# CLUSTER PROFILING - WHAT MAKES EACH CLUSTER UNIQUE?
# ============================================================================

print("\n" + "="*70)
print("CLUSTER PROFILING - ECONOMIC CHARACTERISTICS")
print("="*70)

# Add cluster labels to data
df_profiling = X_combined.copy()
df_profiling['cluster'] = kmeans_labels
df_profiling['bankrupt'] = target.values

# Profile key financial ratios by cluster
key_ratios = [
    'altman_z_score',
    'likviditetsgrad_1',
    'total_gjeldsgrad',
    'egenkapitalandel',
    'driftsmargin',
    'totalkapitalrentabilitet',
    'negativ_egenkapital',
    'log_totalkapital',
]

available_ratios = [r for r in key_ratios if r in df_profiling.columns]

for ratio in available_ratios:
    print(f"\n{ratio}:")
    cluster_means = df_profiling.groupby('cluster')[ratio].mean()
    for cluster_id, mean_val in cluster_means.items():
        bankruptcy_rate = cluster_stats.loc[cluster_id, 'Bankruptcy_Rate']
        print(f"  Cluster {cluster_id} (Bankr: {bankruptcy_rate:.1%}): {mean_val:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = script_dir

# Save cluster assignments
results_df = pd.DataFrame({
    'Orgnr': identifiers['Orgnr'],
    'year': identifiers['year'],
    'bankrupt': target,
    'cluster': kmeans_labels
})
results_df.to_csv(output_dir / 'economic_cluster_assignments.csv', index=False)
print(f"Cluster assignments saved to: economic_cluster_assignments.csv")

# Save PCA results
pca_df = pd.DataFrame(
    X_pca[:, :10],
    columns=[f'PC{i+1}' for i in range(10)]
)
pca_df['Orgnr'] = identifiers['Orgnr'].values
pca_df['year'] = identifiers['year'].values
pca_df['bankrupt'] = target.values
pca_df['cluster'] = kmeans_labels
pca_df.to_csv(output_dir / 'economic_pca_components.csv', index=False)
print(f"PCA components saved to: economic_pca_components.csv")

# Save all PCA loadings
all_loadings = []
for i in range(N_PCA_COMPONENTS):
    loadings = pd.DataFrame({
        'Component': f'PC{i+1}',
        'Feature': feature_names,
        'Loading': pca.components_[i]
    })
    all_loadings.append(loadings)

pd.concat(all_loadings).to_csv(output_dir / 'economic_pca_loadings.csv', index=False)
print(f"PCA loadings saved to: economic_pca_loadings.csv")

# Save summary
summary = {
    'model': 'Economic Fundamentals Unsupervised Model',
    'date': datetime.now().isoformat(),
    'focus': 'Economic distress patterns (excluding filing behavior)',
    'data': {
        'total_observations': len(df_all),
        'bankruptcy_rate': float(target.mean())
    },
    'features': {
        'raw_accounting': len(raw_accounting),
        'financial_ratios': len([f for f in financial_ratios if f in df_all.columns]),
        'temporal_economic': len([f for f in temporal_economic if f in df_all.columns]),
        'warning_signals': len([f for f in warning_signals if f in df_all.columns]),
        'company_characteristics': len([f for f in company_chars if f in df_all.columns]),
        'categorical': len(categorical_features),
        'total': len(feature_names),
        'excluded_missingness_indicators': True
    },
    'pca': {
        'n_components': N_PCA_COMPONENTS,
        'variance_explained': float(cumulative_var[-1])
    },
    'clustering': {
        'best_k': int(best_k),
        'silhouette_score': float(best_silhouette),
        'cluster_sizes': cluster_stats['Companies'].tolist(),
        'cluster_bankruptcy_rates': cluster_stats['Bankruptcy_Rate'].tolist()
    }
}

with open(output_dir / 'economic_fundamentals_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: economic_fundamentals_results.json")

print("\n" + "="*70)
print("ECONOMIC FUNDAMENTALS MODEL COMPLETE")
print("="*70)
print(f"\nKey Achievement:")
print(f"  Discovered bankruptcy patterns based on ECONOMIC HEALTH")
print(f"  NOT based on filing behavior or missing data")
print(f"  {best_k} clusters found with silhouette score {best_silhouette:.4f}")
print(f"  Used {len(feature_names)} economic fundamental features")
