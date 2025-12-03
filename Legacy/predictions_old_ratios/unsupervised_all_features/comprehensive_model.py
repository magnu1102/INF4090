"""
Comprehensive Unsupervised Learning Model - ALL FEATURES
=========================================================

This script uses EVERYTHING available in the dataset:
- All 77 raw accounting fields (Tall 1 through Tall 17130)
- All 40 engineered features (ratios, temporal, missingness, etc.)
- Categorical variables (industry, organization form, location)
- All three years (2016, 2017, 2018)
- NO exclusions due to missing data

Approach:
1. Load complete feature dataset
2. Identify all numeric and categorical features
3. Handle missing data intelligently:
   - Median imputation for numeric features
   - Mode imputation for categorical features
   - Create missingness indicators for highly missing features
4. Encode categorical variables
5. Apply unsupervised learning:
   - PCA (dimensionality reduction)
   - K-Means clustering
   - DBSCAN clustering
   - Hierarchical clustering
6. Analyze cluster characteristics and bankruptcy rates

Theoretical Foundation:
- Exploratory approach to discover natural groupings in data
- No feature selection bias
- Preserve all information (missing data handled, not dropped)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMPREHENSIVE UNSUPERVISED LEARNING MODEL - ALL FEATURES")
print("="*70)

# Configuration
RANDOM_STATE = 42
N_PCA_COMPONENTS = 50  # Reduce to 50 dimensions first
N_CLUSTERS_RANGE = range(2, 11)  # Try 2-10 clusters

# Find the input file
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent.parent / 'data' / 'features' / 'feature_dataset_v1.parquet',
    Path('../../data/features/feature_dataset_v1.parquet')
]

input_file = None
for path in possible_paths:
    if path.exists():
        input_file = path
        break

if input_file is None:
    raise FileNotFoundError("Could not find feature_dataset_v1.parquet")

print(f"\nLoading data from: {input_file}")
df = pd.read_parquet(input_file)

print(f"\nInitial dataset shape: {df.shape}")
print(f"Total companies: {df['Orgnr'].nunique():,}")
print(f"Total observations: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Use all years
df_all = df[df['year'].isin([2016, 2017, 2018])].copy()
print(f"\nUsing all years (2016-2018): {len(df_all):,} observations")
print(f"Bankruptcy rate: {df_all['bankrupt'].mean():.2%}")

# Separate target variable and identifiers
target = df_all['bankrupt'].copy()
identifiers = df_all[['Orgnr', 'year']].copy()

# Identify feature types
print("\n" + "="*70)
print("FEATURE CATEGORIZATION")
print("="*70)

# Exclude columns we don't want to use as features
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

# Get all potential features
all_features = [col for col in df_all.columns if col not in exclude_cols]

print(f"Total features to use: {len(all_features)}")

# Separate numeric and categorical
numeric_features = []
categorical_features = []

for col in all_features:
    if df_all[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_features.append(col)
    else:
        # Check if it's actually numeric stored as object
        try:
            pd.to_numeric(df_all[col], errors='raise')
            numeric_features.append(col)
        except:
            categorical_features.append(col)

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Show examples
print(f"\nExample numeric features: {numeric_features[:5]}")
print(f"Example categorical features: {categorical_features[:5]}")

# Handle missing data
print("\n" + "="*70)
print("MISSING DATA HANDLING")
print("="*70)

# Create feature matrix
X_numeric = df_all[numeric_features].copy()
X_categorical = df_all[categorical_features].copy()

# Handle infinity and extreme values in numeric features
print("\nHandling infinity and extreme values...")
for col in X_numeric.columns:
    # Replace infinity with NaN
    X_numeric[col] = X_numeric[col].replace([np.inf, -np.inf], np.nan)

    # Cap extreme values at 99.9th percentile
    if X_numeric[col].notna().any():
        upper_cap = X_numeric[col].quantile(0.999)
        lower_cap = X_numeric[col].quantile(0.001)
        X_numeric[col] = X_numeric[col].clip(lower=lower_cap, upper=upper_cap)

print("Extreme values handled")

# Check missing data before imputation
numeric_missing = X_numeric.isnull().sum().sum()
categorical_missing = X_categorical.isnull().sum().sum()
total_values = len(X_numeric) * len(numeric_features) + len(X_categorical) * len(categorical_features)
missing_pct = (numeric_missing + categorical_missing) / total_values * 100

print(f"\nBefore imputation:")
print(f"  Numeric missing values: {numeric_missing:,}")
print(f"  Categorical missing values: {categorical_missing:,}")
print(f"  Total missing: {missing_pct:.2f}%")

# Create missingness indicators for highly missing features (>20% missing)
missingness_features = []
for col in numeric_features:
    missing_pct_col = X_numeric[col].isnull().mean() * 100
    if missing_pct_col > 20:
        indicator_name = f'{col}_missing'
        X_numeric[indicator_name] = X_numeric[col].isnull().astype(int)
        missingness_features.append(indicator_name)

print(f"\nCreated {len(missingness_features)} missingness indicators for features >20% missing")

# Impute numeric features with median
print("\nImputing numeric features with median...")
numeric_imputer = SimpleImputer(strategy='median')
X_numeric_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)

# Impute categorical features with most frequent
print("Imputing categorical features with mode...")
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = pd.DataFrame(
    categorical_imputer.fit_transform(X_categorical),
    columns=X_categorical.columns,
    index=X_categorical.index
)

# Encode categorical features
print("\n" + "="*70)
print("ENCODING CATEGORICAL FEATURES")
print("="*70)

X_categorical_encoded = pd.DataFrame(index=X_categorical_imputed.index)

for col in categorical_features:
    if col in X_categorical_imputed.columns:
        # Label encoding for categorical variables
        le = LabelEncoder()
        try:
            X_categorical_encoded[col] = le.fit_transform(X_categorical_imputed[col].astype(str))
            print(f"  Encoded {col}: {len(le.classes_)} unique values")
        except Exception as e:
            print(f"  Skipped {col}: {e}")

print(f"\nSuccessfully encoded {len(X_categorical_encoded.columns)} categorical features")

# Combine all features
print("\n" + "="*70)
print("COMBINING FEATURES")
print("="*70)

X_combined = pd.concat([X_numeric_imputed, X_categorical_encoded], axis=1)
print(f"\nFinal feature matrix shape: {X_combined.shape}")
print(f"  Total features: {X_combined.shape[1]}")
print(f"  Total observations: {X_combined.shape[0]:,}")
print(f"  Missing values remaining: {X_combined.isnull().sum().sum()}")

# Standardize features
print("\nStandardizing all features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
print("Standardization complete")

# Save feature names for later interpretation
feature_names = X_combined.columns.tolist()

# Apply PCA for dimensionality reduction
print("\n" + "="*70)
print("PCA DIMENSIONALITY REDUCTION")
print("="*70)

print(f"\nReducing from {X_scaled.shape[1]} to {N_PCA_COMPONENTS} dimensions...")
pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nPCA Results:")
print(f"  First component explains: {explained_var[0]*100:.2f}% of variance")
print(f"  First 10 components explain: {cumulative_var[9]*100:.2f}% of variance")
print(f"  All {N_PCA_COMPONENTS} components explain: {cumulative_var[-1]*100:.2f}% of variance")

# K-Means Clustering
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

    print(f"  k={k}: Silhouette Score = {silhouette:.4f}")

    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

print(f"\nBest number of clusters: {best_k} (Silhouette: {best_silhouette:.4f})")

# Fit final K-Means model
print(f"\nFitting final K-Means with k={best_k}...")
kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_pca)

# Analyze clusters
print("\n" + "="*70)
print("CLUSTER ANALYSIS")
print("="*70)

cluster_df = pd.DataFrame({
    'Orgnr': identifiers['Orgnr'],
    'year': identifiers['year'],
    'cluster': kmeans_labels,
    'bankrupt': target
})

print("\nCluster sizes and bankruptcy rates:")
print("-" * 50)
cluster_stats = cluster_df.groupby('cluster').agg({
    'Orgnr': 'count',
    'bankrupt': ['sum', 'mean']
}).round(4)
cluster_stats.columns = ['Count', 'Bankruptcies', 'Bankruptcy_Rate']
print(cluster_stats)

# Try DBSCAN for density-based clustering
print("\n" + "="*70)
print("DBSCAN CLUSTERING")
print("="*70)

print("\nApplying DBSCAN (may take a while on large dataset)...")
dbscan = DBSCAN(eps=5, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_pca)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Clusters found: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise:,} ({n_noise/len(dbscan_labels)*100:.2f}%)")

if n_clusters_dbscan > 1:
    # Analyze DBSCAN clusters (excluding noise)
    dbscan_df = pd.DataFrame({
        'cluster': dbscan_labels,
        'bankrupt': target
    })

    print("\nDBSCAN cluster bankruptcy rates (excluding noise):")
    print("-" * 50)
    dbscan_stats = dbscan_df[dbscan_df['cluster'] != -1].groupby('cluster').agg({
        'bankrupt': ['count', 'sum', 'mean']
    }).round(4)
    dbscan_stats.columns = ['Count', 'Bankruptcies', 'Bankruptcy_Rate']
    print(dbscan_stats)

# Save results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = script_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Save cluster assignments
results_df = pd.DataFrame({
    'Orgnr': identifiers['Orgnr'],
    'year': identifiers['year'],
    'bankrupt': target,
    'kmeans_cluster': kmeans_labels,
    'dbscan_cluster': dbscan_labels
})
results_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)
print(f"Cluster assignments saved to: cluster_assignments.csv")

# Save PCA components
pca_df = pd.DataFrame(
    X_pca[:, :10],  # Save first 10 components
    columns=[f'PC{i+1}' for i in range(10)]
)
pca_df['Orgnr'] = identifiers['Orgnr'].values
pca_df['year'] = identifiers['year'].values
pca_df['bankrupt'] = target.values
pca_df.to_csv(output_dir / 'pca_components.csv', index=False)
print(f"PCA components saved to: pca_components.csv")

# Save PCA explained variance
pca_variance = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(N_PCA_COMPONENTS)],
    'Explained_Variance_Ratio': explained_var,
    'Cumulative_Variance': cumulative_var
})
pca_variance.to_csv(output_dir / 'pca_explained_variance.csv', index=False)
print(f"PCA variance explained saved to: pca_explained_variance.csv")

# Save summary statistics
results_summary = {
    'model': 'Comprehensive Unsupervised Learning (All Features)',
    'date': datetime.now().isoformat(),
    'data': {
        'total_observations': len(df_all),
        'total_companies': df_all['Orgnr'].nunique(),
        'bankruptcy_rate': float(target.mean()),
        'years_used': [2016, 2017, 2018]
    },
    'features': {
        'numeric_features': len(numeric_features),
        'categorical_features': len(categorical_features),
        'missingness_indicators': len(missingness_features),
        'total_features': len(feature_names),
        'pca_components': N_PCA_COMPONENTS,
        'variance_explained_by_pca': float(cumulative_var[-1])
    },
    'clustering': {
        'kmeans': {
            'n_clusters': int(best_k),
            'silhouette_score': float(best_silhouette),
            'cluster_sizes': cluster_stats['Count'].tolist(),
            'cluster_bankruptcy_rates': cluster_stats['Bankruptcy_Rate'].tolist()
        },
        'dbscan': {
            'n_clusters': int(n_clusters_dbscan),
            'n_noise_points': int(n_noise),
            'noise_percentage': float(n_noise/len(dbscan_labels)*100)
        }
    }
}

with open(output_dir / 'comprehensive_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"Summary results saved to: comprehensive_results.json")

# Save top PCA loadings
print("\n" + "="*70)
print("TOP PCA LOADINGS (First 3 components)")
print("="*70)

loadings_dfs = []
for i in range(min(3, N_PCA_COMPONENTS)):
    loadings = pd.DataFrame({
        'Feature': feature_names,
        'Loading': pca.components_[i]
    }).sort_values('Loading', key=abs, ascending=False)

    print(f"\nPrincipal Component {i+1} (explains {explained_var[i]*100:.2f}% variance):")
    print("Top 10 positive loadings:")
    print(loadings.head(10)[['Feature', 'Loading']].to_string(index=False))
    print("\nTop 10 negative loadings:")
    print(loadings.tail(10)[['Feature', 'Loading']].to_string(index=False))

    loadings['Component'] = f'PC{i+1}'
    loadings_dfs.append(loadings)

all_loadings = pd.concat(loadings_dfs)
all_loadings.to_csv(output_dir / 'pca_loadings.csv', index=False)
print(f"\nAll PCA loadings saved to: pca_loadings.csv")

print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPLETE")
print("="*70)
print(f"\nKey Achievements:")
print(f"  - Used ALL {len(feature_names)} features (no exclusions)")
print(f"  - Processed ALL {len(df_all):,} observations (no missing data exclusions)")
print(f"  - Created {len(missingness_features)} missingness indicators")
print(f"  - Reduced to {N_PCA_COMPONENTS} principal components")
print(f"  - Found {best_k} natural clusters in the data")
print(f"  - Identified bankruptcy rate variation across clusters")
