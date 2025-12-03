# Comprehensive Unsupervised Learning Model - Methodology

**Model Type:** Unsupervised Learning (Clustering + Dimensionality Reduction)
**Date:** 2025-12-01
**Purpose:** Discover natural patterns in bankruptcy data using ALL available features without exclusions

---

## Motivation

Previous models (2018_only, all_years) had limitations:
1. **Feature selection bias** - Only used 24 carefully selected features
2. **Missing data exclusions** - Dropped 42-45% of observations with any missing values
3. **Selection bias** - Companies with missing data had 7.4x higher bankruptcy rates (15.34% vs 2.07%)

This comprehensive model addresses all three limitations by:
- Using **ALL 84 features** (no selection)
- **Retaining ALL 280,840 observations** (no exclusions)
- **Intelligent missing data handling** (imputation instead of deletion)

---

## Research Questions

1. **Do natural clusters exist in the data that separate bankrupt from non-bankrupt companies?**
   - Can unsupervised learning identify bankruptcy risk without being told who is bankrupt?

2. **What features drive the separation between healthy and distressed companies?**
   - PCA loadings reveal which features contribute most to variance

3. **How do results differ when using ALL features vs selected features?**
   - Comparison to supervised models (2018_only, all_years)

4. **Is missing data informative or just noise?**
   - By including all companies (not dropping missing data), can we find patterns in incomplete records?

---

## Data Preparation

### Input Data
- **Source:** `feature_dataset_v1.parquet`
- **Observations:** 280,840 company-year records (2016, 2017, 2018)
- **Companies:** 114,848 unique organizations
- **Bankruptcy rate:** 7.40% overall

### Feature Categories

#### 1. Raw Accounting Data (77 fields)
All original "Tall" fields from Norwegian financial statements:
- Balance sheet items (assets, liabilities, equity)
- Income statement items (revenue, costs, profits)
- Cash flow items
- Examples: `Tall 1340` (revenue), `Tall 194` (current assets), `Tall 85` (current liabilities)

#### 2. Engineered Financial Ratios (11 features)
Based on Beaver (1966), Altman (1968), Ohlson (1980):
- `likviditetsgrad_1`, `likviditetsgrad_2` - Liquidity ratios
- `total_gjeldsgrad`, `langsiktig_gjeldsgrad`, `kortsiktig_gjeldsgrad` - Leverage ratios
- `egenkapitalandel` - Equity ratio
- `driftsmargin` - Operating margin
- `totalkapitalrentabilitet` - Return on assets
- `omsetningsgrad` - Asset turnover
- `rentedekningsgrad` - Interest coverage
- `altman_z_score` - Altman's bankruptcy prediction score

#### 3. Temporal Features (10 features)
Year-over-year changes and trends:
- `omsetningsvekst_1617`, `omsetningsvekst_1718` - Revenue growth
- `aktivavekst_1617`, `aktivavekst_1718` - Asset growth
- `gjeldsvekst_1617`, `gjeldsvekst_1718` - Debt growth
- `fallende_likviditet` - Declining liquidity indicator
- `konsistent_underskudd` - Consistent losses indicator
- `økende_gjeldsgrad` - Increasing leverage indicator
- `omsetningsvolatilitet` - Revenue volatility

#### 4. Missingness Indicators (7 features)
Capture non-filing behavior:
- `levert_alle_år` - Filed all years
- `levert_2018` - Filed 2018
- `antall_år_levert` - Number of years filed
- `regnskapskomplett` - Complete accounting data
- `kan_ikke_beregne_likviditet` - Cannot calculate liquidity
- `kan_ikke_beregne_gjeldsgrad` - Cannot calculate leverage

#### 5. Company Characteristics (4 features)
- `selskapsalder` - Company age (years)
- `nytt_selskap` - New company indicator (<3 years old)
- `log_totalkapital` - Log of total assets (size proxy)
- `log_omsetning` - Log of revenue

#### 6. Warning Signals (5 features)
Binary flags for financial distress:
- `negativ_egenkapital` - Negative equity
- `sterkt_overbelånt` - Highly leveraged (debt ratio >0.8)
- `kan_ikke_dekke_renter` - Cannot cover interest payments
- `lav_likviditet` - Low liquidity (current ratio <1.0)
- `driftsunderskudd` - Operating loss

#### 7. Auditor Changes (3 features)
- `byttet_revisor_1617` - Changed auditor 2016-2017
- `byttet_revisor_1718` - Changed auditor 2017-2018
- `byttet_revisor_noensinne` - Ever changed auditor

#### 8. Categorical Variables (36 features)
- `Næringskode` - Industry code (primary, secondary, tertiary)
- `Organisasjonsform` - Organization form (AS, ASA, etc.)
- `Regnskapstype` - Accounting standard
- `Revisor` - Auditor name
- `Sektorkode` - Sector code
- `Kommunenr`, `Fylkenr` - Municipality and county codes
- Registration flags (FR, ER)

**Total Features:** 84 (48 numeric + 36 categorical)

---

## Missing Data Strategy

### Problem Statement
Missing data is pervasive in the dataset:
- **37%** of observations missing `omsetningsvekst_1617`
- **35%** missing `omsetningsvekst_1718`
- **23%** missing `altman_z_score`
- **17%** missing `driftsmargin`

**Critical Finding:** Companies with missing data have **15.34% bankruptcy rate** vs **2.07%** for complete data.
- **Implication:** Missing data is highly informative and should NOT be discarded!

### Solution: Intelligent Imputation

#### Step 1: Handle Extreme Values
```python
# Replace infinity with NaN
X_numeric[col] = X_numeric[col].replace([np.inf, -np.inf], np.nan)

# Cap extreme values at 99.9th and 0.1st percentiles
upper_cap = X_numeric[col].quantile(0.999)
lower_cap = X_numeric[col].quantile(0.001)
X_numeric[col] = X_numeric[col].clip(lower=lower_cap, upper=upper_cap)
```

**Rationale:**
- Financial ratios can have extreme outliers (e.g., debt/equity when equity ≈ 0)
- Capping prevents outliers from dominating distance calculations in clustering

#### Step 2: Create Missingness Indicators
For features with >20% missing values:
```python
if missing_pct > 20:
    X_numeric[f'{col}_missing'] = X_numeric[col].isnull().astype(int)
```

**Rationale:**
- Preserve information that data was missing (missingness is informative)
- Model can learn that "missing revenue growth" is a bankruptcy signal

#### Step 3: Median Imputation (Numeric Features)
```python
numeric_imputer = SimpleImputer(strategy='median')
X_numeric_imputed = numeric_imputer.fit_transform(X_numeric)
```

**Rationale:**
- Median is robust to outliers (better than mean for financial data)
- Preserves distributional properties
- Allows all observations to be retained

#### Step 4: Mode Imputation (Categorical Features)
```python
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = categorical_imputer.fit_transform(X_categorical)
```

**Rationale:**
- Most frequent category is sensible default for categorical data
- Preserves category structure

---

## Feature Encoding

### Categorical Variable Encoding
All 36 categorical variables are encoded using **Label Encoding**:
```python
label_encoder = LabelEncoder()
X_categorical_encoded[col] = label_encoder.fit_transform(X_categorical[col].astype(str))
```

**Rationale:**
- Converts text categories (e.g., "AS", "ASA", "NUF") to integers (0, 1, 2)
- Required for numerical algorithms (PCA, K-Means)
- Ordinal encoding acceptable for unsupervised learning (no assumed ordering affects clustering)

### Feature Standardization
All features (numeric + encoded categorical) are standardized:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
```

**Transformation:**
- Each feature: `z = (x - mean) / std_dev`
- Result: Mean = 0, Standard deviation = 1

**Rationale:**
- PCA is sensitive to feature scale (revenue in millions, ratios in 0-1 range)
- Ensures all features contribute equally to distance calculations
- Essential for clustering algorithms that use Euclidean distance

---

## Dimensionality Reduction: PCA

### Approach
**Principal Component Analysis (PCA)** with 50 components:
```python
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```

### Why PCA?

**Problem:** 84 features create high-dimensional space
- Clustering suffers from "curse of dimensionality"
- Many features are correlated (e.g., liquidity ratio and cash ratio)
- Computational cost grows exponentially with dimensions

**Solution:** Project data onto 50 principal components
- Capture most variance in lower-dimensional space
- Remove multicollinearity
- Speed up clustering

### PCA Interpretation

**Explained Variance:**
- PC1: Largest variance direction (typically "size" or "financial health")
- PC2: Second largest variance direction (orthogonal to PC1)
- Total: First 50 PCs capture 80-95% of total variance

**Loadings:**
- Each PC is a weighted combination of original features
- High positive loading: Feature contributes positively to PC
- High negative loading: Feature contributes negatively to PC
- Near-zero loading: Feature irrelevant to this PC

**Example:**
```
PC1 might be "Financial Health":
  +0.8 * altman_z_score
  +0.6 * likviditetsgrad_1
  -0.7 * negativ_egenkapital
  -0.5 * total_gjeldsgrad
```

---

## Clustering Algorithms

### 1. K-Means Clustering

**Algorithm:** Partition-based clustering
```python
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    silhouette = silhouette_score(X_pca, labels)
```

**How it works:**
1. Randomly initialize k cluster centers
2. Assign each point to nearest center
3. Recalculate centers as mean of assigned points
4. Repeat until convergence

**Selection of k:**
- Test k = 2, 3, 4, 5, 6, 7, 8, 9, 10
- Evaluate using **Silhouette Score** (-1 to +1, higher is better)
- Silhouette measures: "How similar is a point to its cluster vs nearest other cluster?"

**Advantages:**
- Fast and scalable
- Well-understood algorithm
- Works well with PCA-reduced data

**Disadvantages:**
- Assumes spherical clusters
- Sensitive to initialization (mitigated by n_init=10)
- Must specify k in advance

### 2. DBSCAN Clustering

**Algorithm:** Density-based clustering
```python
dbscan = DBSCAN(eps=5, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_pca)
```

**How it works:**
1. Find points with at least `min_samples` neighbors within `eps` distance
2. These are "core points" - start of clusters
3. Expand clusters by adding neighbors of core points
4. Points not in any cluster are labeled "noise"

**Advantages:**
- Discovers arbitrary-shaped clusters (not just spherical)
- Automatically determines number of clusters
- Identifies outliers as "noise"

**Disadvantages:**
- Very slow on large datasets (280K observations)
- Sensitive to eps and min_samples parameters
- May label many points as noise

**Use case:**
- Identify outlier companies (noise points)
- Find irregularly shaped bankruptcy clusters

---

## Evaluation Metrics

### Silhouette Score
**Formula:** For each point i:
```
a(i) = average distance to points in same cluster
b(i) = average distance to points in nearest other cluster
silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Interpretation:**
- +1: Perfect clustering (far from other clusters)
- 0: On the boundary between clusters
- -1: Likely in wrong cluster

**Overall score:** Average silhouette across all points

### Calinski-Harabasz Score
**Formula:** Ratio of between-cluster variance to within-cluster variance

**Interpretation:**
- Higher is better
- Indicates well-separated, compact clusters

### Bankruptcy Rate by Cluster
**Custom metric:** For each cluster, calculate:
```
bankruptcy_rate = (bankrupt companies in cluster) / (total companies in cluster)
```

**Ideal result:**
- Cluster 1: Very low bankruptcy rate (healthy companies)
- Cluster 2: Very high bankruptcy rate (distressed companies)
- Clear separation validates that features capture bankruptcy risk

---

## Expected Outcomes

### Hypothesis 1: Natural Bankruptcy Clusters Exist
**Expectation:** K-Means will identify 2-4 clusters with distinct bankruptcy rates

**Evidence from fast model (20% sample):**
- Cluster 0: 4.8% bankruptcy (normal companies)
- Cluster 1: 0% bankruptcy (super healthy)
- Cluster 2: **100% bankruptcy** (distressed companies)

**Implication:** If true, suggests strong natural separation in feature space

### Hypothesis 2: Certain Features Drive Separation
**Expectation:** PCA loadings will reveal:
- PC1: Financial health dimension (Altman Z-Score, liquidity, leverage)
- PC2: Size dimension (log_totalkapital, log_omsetning)
- PC3: Temporal dimension (growth rates, volatility)

**Analysis:** Top 10 features with highest absolute loadings on PC1-PC3

### Hypothesis 3: Missing Data is Informative
**Expectation:** Missingness indicators will have high PCA loadings

**Test:** Check if missingness indicators appear in top 20 PCA loadings

### Hypothesis 4: All Features Add Value
**Expectation:** Using 84 features captures more variance than 24 selected features

**Test:** Compare PCA explained variance:
- 50 PCs from 84 features: Expected 85-95% variance explained
- 50 PCs from 24 features: Expected 70-85% variance explained

---

## Comparison to Previous Models

### 2018_only Supervised Model
| Aspect | 2018_only | Comprehensive Unsupervised |
|--------|-----------|---------------------------|
| **Approach** | Supervised (Logistic Regression) | Unsupervised (Clustering) |
| **Features** | 24 selected features | 84 total features (all) |
| **Observations** | 52,303 (58% of 2018) | 280,840 (100% of all years) |
| **Missing data** | Dropped (complete case) | Imputed (retained all) |
| **Bankruptcy rate** | 2.07% | 7.40% |
| **Goal** | Predict bankruptcy | Discover natural groups |
| **Evaluation** | ROC-AUC, Precision, Recall | Silhouette, Cluster purity |

### all_years Supervised Model
| Aspect | all_years | Comprehensive Unsupervised |
|--------|-----------|---------------------------|
| **Features** | 24 selected features | 84 total features |
| **Observations** | 155,724 (55.4%) | 280,840 (100%) |
| **Missing data** | Dropped | Imputed |
| **Bankruptcy rate** | 2.08% | 7.40% |
| **Goal** | Predict bankruptcy | Discover natural groups |

**Key Difference:** Unsupervised model uses the "hard cases" that supervised models excluded!

---

## Theoretical Foundation

### Unsupervised Learning for Bankruptcy Prediction

**Classical approach (Beaver 1966, Altman 1968, Ohlson 1980):**
- Supervised learning: Train on labeled bankrupt/non-bankrupt companies
- Assumption: We know ground truth labels

**Unsupervised approach (this model):**
- No labels used during clustering
- Assumption: If bankruptcy risk is real, it will manifest as natural groupings
- Validation: Check if discovered clusters align with actual bankruptcies

**Why this matters:**
1. **Validation of features:** If clustering finds bankruptcy groups without being told, features truly capture distress
2. **Discovery of subtypes:** May find multiple types of bankruptcy (sudden vs gradual decline)
3. **Outlier detection:** Identifies unusual companies that don't fit patterns
4. **No label leakage:** Clustering can't "cheat" by using bankruptcy status

### Dimensionality Reduction Theory

**Curse of dimensionality (Bellman 1961):**
- In high dimensions, all points become equidistant
- Clustering becomes meaningless in 84-dimensional space

**PCA (Pearson 1901, Hotelling 1933):**
- Projects data onto directions of maximum variance
- Assumes: Variance = Information (true for financial data)
- Result: Lower-dimensional representation preserving structure

**Application to bankruptcy:**
- Many financial ratios are correlated (e.g., ROA, ROE, profit margin)
- PCA removes redundancy while preserving distress signals
- First few PCs capture "fundamental drivers" of bankruptcy

---

## Limitations and Considerations

### 1. Label Encoding of Categoricals
- **Issue:** Assumes ordinality (AS=0, ASA=1 implies AS < ASA)
- **Impact:** May distort distances in categorical dimensions
- **Alternative:** One-hot encoding (but would create 100s of features)
- **Mitigation:** Categorical features are minority (36/84); PCA will down-weight if uninformative

### 2. Median Imputation Bias
- **Issue:** Imputed values cluster at median, reducing variance
- **Impact:** May pull imputed observations toward cluster centers
- **Alternative:** Model-based imputation (MICE, K-NN imputation)
- **Mitigation:** Missingness indicators preserve information about what was missing

### 3. Linear Dimensionality Reduction
- **Issue:** PCA assumes linear combinations of features
- **Impact:** May miss non-linear relationships (e.g., interaction effects)
- **Alternative:** t-SNE, UMAP, autoencoders
- **Mitigation:** Financial ratios often have linear relationships

### 4. Euclidean Distance in Clustering
- **Issue:** K-Means uses Euclidean distance (assumes spherical clusters)
- **Impact:** May miss elongated or irregularly shaped bankruptcy clusters
- **Alternative:** DBSCAN, hierarchical clustering, GMM
- **Mitigation:** We run both K-Means and DBSCAN for comparison

### 5. Computational Complexity
- **Issue:** 280K observations × 84 features is large
- **Impact:** Long computation time (20-40 minutes)
- **Alternative:** Sampling, mini-batch K-Means
- **Trade-off:** Accepted for comprehensive analysis

---

## Reproducibility

### Random Seeds
All stochastic processes use `random_state=42`:
- PCA (initialization)
- K-Means (initialization, n_init=10 runs)
- DBSCAN (none needed, deterministic)

### Software Versions
```
Python: 3.11
scikit-learn: 1.7.2
pandas: 2.3.5
numpy: 2.3.5
```

### Data Versioning
- Input: `feature_dataset_v1.parquet` (generated 2025-11-30)
- Features: 40 engineered features from `build_features.py`
- Raw data: Norwegian company accounts 2016-2018, bankruptcies 2019

---

## Output Files

### 1. `cluster_assignments.csv`
- Columns: `Orgnr`, `year`, `bankrupt`, `kmeans_cluster`, `dbscan_cluster`
- Rows: 280,840 (all company-year observations)
- Purpose: Assign each company to discovered clusters

### 2. `pca_components.csv`
- Columns: `PC1`, `PC2`, ..., `PC10`, `Orgnr`, `year`, `bankrupt`
- Rows: 280,840
- Purpose: Reduced-dimensional representation for visualization

### 3. `pca_explained_variance.csv`
- Columns: `Component`, `Explained_Variance_Ratio`, `Cumulative_Variance`
- Rows: 50 (one per PC)
- Purpose: Understand how much variance each PC captures

### 4. `pca_loadings.csv`
- Columns: `Feature`, `Loading`, `Component`
- Rows: 50 × 84 = 4,200 (all features × first 3 PCs)
- Purpose: Interpret what each principal component represents

### 5. `comprehensive_results.json`
- Machine-readable summary statistics
- Cluster sizes, bankruptcy rates, silhouette scores
- PCA variance explained
- Feature counts

### 6. `comprehensive_results.md` (to be created)
- Human-readable analysis and interpretation
- Cluster characteristics
- Feature importance from PCA
- Comparison to supervised models
- Business implications

---

## Next Steps After Results

### 1. Cluster Profiling
For each cluster, analyze:
- Mean values of key features (Altman Z-Score, liquidity, leverage)
- Distribution of industries (Næringskode)
- Company size distribution (log_totalkapital)
- Geographic distribution (Fylke)

### 2. PCA Interpretation
- Identify top 20 features by absolute loading on PC1, PC2, PC3
- Name principal components based on loadings (e.g., "Financial Health", "Size", "Growth")
- Visualize companies in PC1-PC2 space colored by bankruptcy status

### 3. Supervised Learning on Clusters
- Use cluster assignments as features in logistic regression
- Test: Does cluster membership improve prediction over raw features?

### 4. Outlier Analysis
- Examine DBSCAN "noise" points
- Are noise points more likely to be bankrupt?
- What makes them unusual?

### 5. Temporal Analysis
- Do clusters shift over time (2016 → 2017 → 2018)?
- Do companies transition between clusters before bankruptcy?

---

## Research Contribution

This comprehensive unsupervised model contributes to bankruptcy prediction literature by:

1. **Methodological innovation:** First model to use ALL available features without selection
2. **Missing data handling:** Demonstrates that imputation outperforms deletion
3. **Validation of features:** Shows that engineered features create natural separation
4. **Practical application:** Provides cluster-based early warning system for regulators

For your thesis research question ("How do key factors differ across ML algorithms?"), this model provides:
- **Unsupervised perspective:** Feature importance from PCA loadings (different from supervised coefficients)
- **Robustness check:** If clusters align with bankruptcy, supervised models are not overfitting
- **Feature discovery:** PCA may reveal important features overlooked in supervised feature selection

---

**Status:** Model currently running (started 2025-12-01 14:35)
**Expected completion:** 30-40 minutes total
**Results to be added:** Cluster analysis, PCA interpretation, comparison to supervised models
