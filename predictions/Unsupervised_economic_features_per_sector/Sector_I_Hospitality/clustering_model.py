"""
Unsupervised Learning: Pure Economic Features - Sector I (Hospitality)
=======================================================================

NACE Codes: 55-56 (Accommodation and Food Service Activities)

Features Used:
- ALL raw accounting numbers (Tall fields)
- ALL financial ratios (10 ratios)
- NO temporal features (let model discover patterns naturally)
- NO filing behavior features
- NO company characteristics
- NO missingness indicators

Strategy: Complete case analysis - focus on pure economics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import joblib
from multiprocessing import cpu_count

print("="*80)
print("UNSUPERVISED CLUSTERING: SECTOR I (HOSPITALITY) - PURE ECONOMIC FEATURES")
print("="*80)
print(f"Started: {datetime.now()}")
print(f"CPU cores available: {cpu_count()}")
print(f"Using ALL available computational resources")

RANDOM_STATE = 42

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/7] Loading feature dataset...")
data_dir = Path(r'C:\Users\magnu\Desktop\AI Management\INF4090\data\features')
df = pd.read_parquet(data_dir / 'feature_dataset_v1.parquet')

print(f"  Total observations: {len(df):,}")
print(f"  Total companies: {df['Orgnr'].nunique():,}")

# Filter to all years (2016, 2017, 2018)
df_all = df[df['year'].isin([2016, 2017, 2018])].copy()
print(f"  Observations 2016-2018: {len(df_all):,}")

# Extract NACE code and filter to Sector I (Hospitality: 55-56)
def extract_nace_code(naringskode):
    try:
        code_str = str(naringskode).split('.')[0]
        if code_str and code_str.strip() and code_str.strip()[0].isdigit():
            return int(code_str[:2]) if len(code_str) >= 2 else None
        return None
    except:
        return None

df_all['nace_code'] = df_all['Næringskode'].apply(extract_nace_code)
sector_df = df_all[(df_all['nace_code'] >= 55) & (df_all['nace_code'] <= 56)].copy()

print(f"\nSector I (Hospitality) - NACE 55-56:")
print(f"  Total observations: {len(sector_df):,}")
print(f"  Unique companies: {sector_df['Orgnr'].nunique():,}")
print(f"  Bankruptcy rate: {sector_df['bankrupt'].mean():.2%}")
print(f"  Year distribution:")
for year in [2016, 2017, 2018]:
    year_count = len(sector_df[sector_df['year'] == year])
    print(f"    {year}: {year_count:,} observations")

# ============================================================================
# FEATURE SELECTION: PURE ECONOMIC FEATURES
# ============================================================================

print("\n[2/7] Selecting pure economic features...")

# 1. RAW ACCOUNTING DATA (Tall fields)
raw_accounting = [
    'Tall 1340',   # Salgsinntekt
    'Tall 7709',   # Annen driftsinntekt
    'Tall 72',     # Sum inntekter
    'Tall 146',    # Driftsresultat
    'Tall 217',    # Sum anleggsmidler
    'Tall 194',    # Sum omløpsmidler
    'Tall 85',     # Sum kortsiktig gjeld
    'Tall 86',     # Sum langsiktig gjeld (if exists)
    'Tall 17130',  # Sum finanskostnader
]

# Filter to existing columns
raw_accounting = [col for col in raw_accounting if col in sector_df.columns]

# 2. FINANCIAL RATIOS (all 10 ratios)
financial_ratios = [
    'likviditetsgrad_1',
    'total_gjeldsgrad',
    'langsiktig_gjeldsgrad',
    'kortsiktig_gjeldsgrad',
    'egenkapitalandel',
    'driftsmargin',
    'driftsrentabilitet',
    'omsetningsgrad',
    'rentedekningsgrad',
    'altman_z_score',
]

# Filter to existing columns
financial_ratios = [col for col in financial_ratios if col in sector_df.columns]

# Combine all economic features
feature_columns = raw_accounting + financial_ratios

print(f"\nEconomic features selected:")
print(f"  Raw accounting data: {len(raw_accounting)} features")
for col in raw_accounting:
    print(f"    - {col}")
print(f"  Financial ratios: {len(financial_ratios)} features")
for col in financial_ratios:
    print(f"    - {col}")
print(f"  TOTAL: {len(feature_columns)} features")

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n[3/7] Preparing data...")

# Select features and target
X = sector_df[feature_columns].copy()
y = sector_df['bankrupt'].copy()
orgnr = sector_df['Orgnr'].copy()
year = sector_df['year'].copy()

print(f"Feature matrix shape: {X.shape}")

# Convert all to numeric
print("Converting to numeric types...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle infinity
print("Handling infinity values...")
for col in X.columns:
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)

# Complete case analysis
print("\nComplete case analysis:")
missing_mask = X.isnull().any(axis=1)
n_missing = missing_mask.sum()
n_complete = (~missing_mask).sum()

print(f"  Observations with missing data: {n_missing:,} ({n_missing/len(X)*100:.1f}%)")
print(f"  Complete cases: {n_complete:,} ({n_complete/len(X)*100:.1f}%)")

# Keep only complete cases
X_complete = X[~missing_mask].copy()
y_complete = y[~missing_mask].copy()
orgnr_complete = orgnr[~missing_mask].copy()
year_complete = year[~missing_mask].copy()

print(f"\nFinal dataset:")
print(f"  Observations: {len(X_complete):,}")
print(f"  Features: {X_complete.shape[1]}")
print(f"  Unique companies: {orgnr_complete.nunique():,}")
print(f"  Bankruptcy rate: {y_complete.mean():.2%}")
print(f"  Bankruptcies: {y_complete.sum():,}")

# ============================================================================
# STANDARDIZE FEATURES
# ============================================================================

print("\n[4/7] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_complete)

print(f"  Features scaled to mean=0, std=1")
print(f"  Shape: {X_scaled.shape}")

# ============================================================================
# DIMENSIONALITY REDUCTION (PCA)
# ============================================================================

print("\n[5/7] Performing PCA for dimensionality reduction...")

# Use enough components to explain 95% variance
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

n_components = pca.n_components_
variance_explained = pca.explained_variance_ratio_

print(f"  PCA Components: {n_components}")
print(f"  Total variance explained: {variance_explained.sum():.1%}")
print(f"\nTop 10 components:")
for i in range(min(10, n_components)):
    print(f"    PC{i+1}: {variance_explained[i]:.2%} variance")

# Interpret principal components
print("\n  Interpreting Principal Components...")
components_df = pd.DataFrame(
    pca.components_[:10],  # Top 10 PCs
    columns=feature_columns,
    index=[f'PC{i+1}' for i in range(min(10, n_components))]
)

for i in range(min(5, n_components)):
    pc = components_df.iloc[i]
    top_features = pc.abs().nlargest(5)
    print(f"\n  PC{i+1} top loadings:")
    for feat, loading in top_features.items():
        direction = "+" if pc[feat] > 0 else "-"
        print(f"    {direction} {feat}: {abs(loading):.3f}")

# ============================================================================
# CLUSTERING: K-MEANS
# ============================================================================

print("\n[6/7] Performing K-Means clustering...")
print("  Testing K=2 to K=10 clusters...")
print("  Using ALL CPU cores for parallel processing...")

kmeans_results = []

for k in range(2, 11):
    print(f"\n  K={k}...")

    # KMeans automatically uses multiple cores in newer sklearn versions
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=20,  # More initializations for better results
        max_iter=500,
        random_state=RANDOM_STATE
    )

    cluster_labels = kmeans.fit_predict(X_pca)

    # Calculate metrics
    silhouette = silhouette_score(X_pca, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
    calinski = calinski_harabasz_score(X_pca, cluster_labels)

    # Analyze bankruptcy rates per cluster
    cluster_bankruptcy = pd.DataFrame({
        'cluster': cluster_labels,
        'bankrupt': y_complete.values
    })

    cluster_stats = cluster_bankruptcy.groupby('cluster').agg({
        'bankrupt': ['count', 'sum', 'mean']
    })
    cluster_stats.columns = ['count', 'bankruptcies', 'bankruptcy_rate']

    print(f"    Silhouette: {silhouette:.4f}")
    print(f"    Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"    Calinski-Harabasz: {calinski:.2f}")

    # Find clusters with extreme bankruptcy rates
    max_bankr_cluster = cluster_stats['bankruptcy_rate'].idxmax()
    max_bankr_rate = cluster_stats.loc[max_bankr_cluster, 'bankruptcy_rate']
    min_bankr_cluster = cluster_stats['bankruptcy_rate'].idxmin()
    min_bankr_rate = cluster_stats.loc[min_bankr_cluster, 'bankruptcy_rate']

    print(f"    Bankruptcy rates: {min_bankr_rate:.2%} (cluster {min_bankr_cluster}) to {max_bankr_rate:.2%} (cluster {max_bankr_cluster})")

    kmeans_results.append({
        'k': k,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski,
        'max_bankruptcy_rate': max_bankr_rate,
        'min_bankruptcy_rate': min_bankr_rate,
        'model': kmeans,
        'labels': cluster_labels,
        'cluster_stats': cluster_stats
    })

# Select best K based on silhouette score
best_k_idx = max(range(len(kmeans_results)), key=lambda i: kmeans_results[i]['silhouette'])
best_k = kmeans_results[best_k_idx]['k']
best_silhouette = kmeans_results[best_k_idx]['silhouette']

print(f"\n  Best K: {best_k} (Silhouette: {best_silhouette:.4f})")

# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================

print("\n[7/7] Performing DBSCAN clustering...")
print("  Testing multiple epsilon values...")
print("  Using ALL CPU cores...")

# Test different epsilon values
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
dbscan_results = []

for eps in eps_values:
    print(f"\n  Epsilon={eps}...")

    dbscan = DBSCAN(
        eps=eps,
        min_samples=5,
        metric='euclidean',
        n_jobs=-1  # Use ALL CPU cores
    )

    cluster_labels = dbscan.fit_predict(X_pca)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points: {n_noise:,} ({n_noise/len(cluster_labels)*100:.1f}%)")

    if n_clusters > 1:
        # Only calculate silhouette if we have multiple clusters
        non_noise_mask = cluster_labels != -1
        if non_noise_mask.sum() > 0:
            silhouette = silhouette_score(X_pca[non_noise_mask], cluster_labels[non_noise_mask])
            print(f"    Silhouette (non-noise): {silhouette:.4f}")
        else:
            silhouette = None
    else:
        silhouette = None

    # Analyze bankruptcy rates
    cluster_bankruptcy = pd.DataFrame({
        'cluster': cluster_labels,
        'bankrupt': y_complete.values
    })

    cluster_stats = cluster_bankruptcy.groupby('cluster').agg({
        'bankrupt': ['count', 'sum', 'mean']
    })
    cluster_stats.columns = ['count', 'bankruptcies', 'bankruptcy_rate']

    dbscan_results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'labels': cluster_labels,
        'cluster_stats': cluster_stats
    })

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_dir = Path(__file__).parent

# Save best K-Means model
best_kmeans = kmeans_results[best_k_idx]
best_labels = best_kmeans['labels']

results_df = pd.DataFrame({
    'Orgnr': orgnr_complete.values,
    'year': year_complete.values,
    'bankrupt': y_complete.values,
    'cluster_kmeans': best_labels,
})

# Add original features
for col in feature_columns:
    results_df[col] = X_complete[col].values

results_df.to_csv(output_dir / 'cluster_results.csv', index=False)
print(f"[OK] Saved: cluster_results.csv")

# Save PCA-transformed data
pca_df = pd.DataFrame(
    X_pca[:, :10],  # Top 10 PCs
    columns=[f'PC{i+1}' for i in range(min(10, X_pca.shape[1]))]
)
pca_df['Orgnr'] = orgnr_complete.values
pca_df['year'] = year_complete.values
pca_df['bankrupt'] = y_complete.values
pca_df['cluster'] = best_labels
pca_df.to_csv(output_dir / 'pca_coordinates.csv', index=False)
print(f"[OK] Saved: pca_coordinates.csv")

# Save models
joblib.dump(scaler, output_dir / 'scaler.pkl')
joblib.dump(pca, output_dir / 'pca_model.pkl')
joblib.dump(best_kmeans['model'], output_dir / 'kmeans_model.pkl')
print(f"[OK] Saved: scaler.pkl, pca_model.pkl, kmeans_model.pkl")

# Save cluster statistics
cluster_stats_detailed = []
for cluster_id in range(best_k):
    cluster_mask = best_labels == cluster_id
    cluster_data = X_complete[cluster_mask]

    stats = {
        'cluster': cluster_id,
        'n_observations': cluster_mask.sum(),
        'n_bankruptcies': y_complete[cluster_mask].sum(),
        'bankruptcy_rate': y_complete[cluster_mask].mean(),
    }

    # Add mean values for each feature
    for col in feature_columns:
        stats[f'{col}_mean'] = cluster_data[col].mean()

    cluster_stats_detailed.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats_detailed)
cluster_stats_df.to_csv(output_dir / 'cluster_statistics.csv', index=False)
print(f"[OK] Saved: cluster_statistics.csv")

# Save summary results
summary = {
    'sector': 'I (Hospitality)',
    'nace_codes': '55-56',
    'n_observations': len(X_complete),
    'n_companies': orgnr_complete.nunique(),
    'n_features': len(feature_columns),
    'n_bankruptcies': int(y_complete.sum()),
    'bankruptcy_rate': float(y_complete.mean()),
    'pca_components': int(n_components),
    'variance_explained': float(variance_explained.sum()),
    'best_k': int(best_k),
    'best_silhouette': float(best_silhouette),
    'feature_list': feature_columns,
}

import json
with open(output_dir / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Saved: analysis_summary.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Completed: {datetime.now()}")
print(f"\nBest clustering: K={best_k} (Silhouette: {best_silhouette:.4f})")
print(f"Results saved to: {output_dir}")
