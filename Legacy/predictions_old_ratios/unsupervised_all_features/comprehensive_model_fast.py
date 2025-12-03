"""
Fast Comprehensive Model - Sampled Version
===========================================

Same as comprehensive_model.py but uses a 20% sample for faster execution.
This allows us to test the approach and get results quickly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FAST COMPREHENSIVE MODEL - 20% SAMPLE")
print("="*70)

RANDOM_STATE = 42
SAMPLE_FRAC = 0.2  # Use 20% of data for speed
N_PCA_COMPONENTS = 30
N_CLUSTERS_RANGE = range(2, 8)

# Load data
script_dir = Path(__file__).parent
input_file = script_dir.parent.parent / 'data' / 'features' / 'feature_dataset_v1.parquet'

print(f"\nLoading data from: {input_file}")
df = pd.read_parquet(input_file)

# Sample for speed
df_sampled = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
print(f"\nSampled {len(df_sampled):,} observations ({SAMPLE_FRAC*100}% of {len(df):,})")

target = df_sampled['bankrupt'].copy()
identifiers = df_sampled[['Orgnr', 'year']].copy()

# Feature selection
exclude_cols = ['Orgnr', 'year', 'bankrupt', 'Navn', 'Forretningsadresse',
                'Fadr postnr', 'Fadr poststed', 'Postadresse', 'Padr postnr',
                'Padr poststed', 'Telefon', 'Mobil', 'E-postadresse',
                'Internettadresse', 'Referanse', 'Referanses adresse',
                'Referanses postnr', 'Referanses poststed', 'Styrets leder',
                'Styreleders adresse', 'Styreleders postnr', 'Styreleders poststed',
                'Revisors navn', 'Revisors adresse', 'Revisors postnr',
                'Revisors poststed', 'Regnskapsf�rer', 'Regnskapsf�rers navn',
                'Regnskapsf�rers adresse', 'Regnskapsf�rers postnr',
                'Regnskapsf�rers poststed', 'Siste godkjente �rsregnskap',
                'Konkurs', 'Oppl�st', 'Slettedato, ER', 'Stiftelsesdato',
                'Kapital', 'MVA reg', 'FRIV', 'Antall BEDR', 'Rolletype',
                'Regnskap fra', 'Regnskap til']

all_features = [col for col in df_sampled.columns if col not in exclude_cols]

numeric_features = []
categorical_features = []

for col in all_features:
    if df_sampled[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_features.append(col)
    else:
        try:
            pd.to_numeric(df_sampled[col], errors='raise')
            numeric_features.append(col)
        except:
            categorical_features.append(col)

print(f"\nFeatures: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

# Prepare features
X_numeric = df_sampled[numeric_features].copy()
X_categorical = df_sampled[categorical_features].copy()

# Handle infinity and extremes
print("\nHandling infinity and extreme values...")
for col in X_numeric.columns:
    X_numeric[col] = X_numeric[col].replace([np.inf, -np.inf], np.nan)
    if X_numeric[col].notna().any():
        upper_cap = X_numeric[col].quantile(0.999)
        lower_cap = X_numeric[col].quantile(0.001)
        X_numeric[col] = X_numeric[col].clip(lower=lower_cap, upper=upper_cap)

# Impute
print("Imputing missing values...")
numeric_imputer = SimpleImputer(strategy='median')
X_numeric_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)

categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = pd.DataFrame(
    categorical_imputer.fit_transform(X_categorical),
    columns=X_categorical.columns,
    index=X_categorical.index
)

# Encode categoricals
print("Encoding categorical features...")
X_categorical_encoded = pd.DataFrame(index=X_categorical_imputed.index)

for col in categorical_features:
    if col in X_categorical_imputed.columns:
        le = LabelEncoder()
        try:
            X_categorical_encoded[col] = le.fit_transform(X_categorical_imputed[col].astype(str))
        except:
            pass

print(f"Encoded {len(X_categorical_encoded.columns)} categorical features")

# Combine
X_combined = pd.concat([X_numeric_imputed, X_categorical_encoded], axis=1)
print(f"\nFinal feature matrix: {X_combined.shape}")

# Standardize
print("Standardizing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# PCA
print(f"\nRunning PCA ({N_PCA_COMPONENTS} components)...")
pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"First component: {explained_var[0]*100:.2f}% variance")
print(f"All components: {cumulative_var[-1]*100:.2f}% variance")

# K-Means
print("\nK-Means clustering...")
best_k = 2
best_silhouette = -1

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
cluster_stats.columns = ['Count', 'Bankruptcies', 'Bankruptcy_Rate']
print(cluster_stats)

# Save results
output_dir = script_dir
results_df = pd.DataFrame({
    'Orgnr': identifiers['Orgnr'],
    'year': identifiers['year'],
    'bankrupt': target,
    'cluster': kmeans_labels
})
results_df.to_csv(output_dir / 'fast_cluster_assignments.csv', index=False)

results_summary = {
    'model': 'Fast Comprehensive Model (20% sample)',
    'date': datetime.now().isoformat(),
    'data': {
        'sample_size': len(df_sampled),
        'sample_fraction': SAMPLE_FRAC,
        'bankruptcy_rate': float(target.mean())
    },
    'features': {
        'numeric': len(numeric_features),
        'categorical': len(X_categorical_encoded.columns),
        'total': len(X_combined.columns)
    },
    'pca': {
        'n_components': N_PCA_COMPONENTS,
        'variance_explained': float(cumulative_var[-1])
    },
    'clustering': {
        'best_k': int(best_k),
        'silhouette_score': float(best_silhouette),
        'cluster_sizes': cluster_stats['Count'].tolist(),
        'cluster_bankruptcy_rates': cluster_stats['Bankruptcy_Rate'].tolist()
    }
}

with open(output_dir / 'fast_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved!")
print("="*70)
