"""
All Years Bankruptcy Prediction Model
======================================

This script builds a logistic regression model to predict bankruptcy
using ALL THREE YEARS (2016, 2017, 2018) of company data.

Difference from 2018_only model:
- Uses all company-year observations (3x more data)
- Captures temporal dynamics across multiple years
- Each company can have up to 3 rows (one per year)

Theoretical Foundation:
- Beaver (1966): Working capital ratios and cash flow
- Altman (1968): Z-Score multivariate model
- Ohlson (1980): Logistic regression approach

Model Approach:
- Use all years 2016, 2017, 2018
- Logistic Regression with class weights to handle imbalance
- Stratified train/test split (80/20)
- Feature selection based on theory and completeness
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Find the input file
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent.parent / 'data' / 'features' / 'feature_dataset_v1.parquet',
    Path('data/features/feature_dataset_v1.parquet'),
    Path('../../data/features/feature_dataset_v1.parquet')
]

input_file = None
for path in possible_paths:
    if path.exists():
        input_file = path
        break

if input_file is None:
    raise FileNotFoundError("Could not find feature_dataset_v1.parquet")

print("="*60)
print("ALL YEARS BANKRUPTCY PREDICTION MODEL")
print("="*60)
print(f"\nLoading data from: {input_file}")
df = pd.read_parquet(input_file)

print(f"\nDataset shape: {df.shape}")
print(f"Total companies: {df['Orgnr'].nunique()}")
print(f"Total company-year observations: {len(df)}")

# Use ALL years (2016, 2017, 2018) - this is the key difference
df_all_years = df[df['year'].isin([2016, 2017, 2018])].copy()
print(f"\nAll years data shape: {df_all_years.shape}")
print(f"Year distribution:")
print(df_all_years['year'].value_counts().sort_index())
print(f"\nOverall bankruptcy rate: {df_all_years['bankrupt'].mean():.2%}")
print(f"Bankruptcy rate by year:")
print(df_all_years.groupby('year')['bankrupt'].mean())

# Select features for model
# Using same features as 2018_only model for comparison
feature_columns = [
    # Financial ratios (Beaver, Altman)
    'likviditetsgrad_1',
    'likviditetsgrad_2',
    'total_gjeldsgrad',
    'egenkapitalandel',
    'rentedekningsgrad',
    'driftsmargin',
    'egenkapitalrentabilitet',
    'totalkapitalrentabilitet',
    'kontantstrøm_margin',

    # Altman Z-Score
    'altman_z_score',

    # Temporal features (growth and trends)
    'omsetningsvekst_1617',
    'omsetningsvekst_1718',
    'fallende_likviditet',
    'forverret_lønnsomhet',
    'volatil_omsetning',
    'konsistent_underskudd',

    # Missingness indicators (highly predictive)
    'levert_alle_år',
    'levert_2018',
    'levert_2017',
    'levert_2016',
    'regnskapskomplett',
    'manglende_felt_andel',

    # Company characteristics
    'selskapsalder',
    'nytt_selskap',
    'log_totalkapital',
    'under_bransjemedian_størrelse',

    # Warning signals
    'negativ_egenkapital',
    'sterkt_overbelånt',
    'lav_likviditet',
    'negativ_kontantstrøm',
    'tap_siste_år',

    # Auditor changes
    'byttet_revisor_1617',
    'byttet_revisor_1718',
    'byttet_revisor_noensinne'
]

# Check which features exist in the dataset
available_features = [f for f in feature_columns if f in df_all_years.columns]
missing_features = [f for f in feature_columns if f not in df_all_years.columns]

print(f"\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)
print(f"Available features: {len(available_features)}/{len(feature_columns)}")
if missing_features:
    print(f"Missing features: {missing_features}")

# Prepare X and y
X = df_all_years[available_features].copy()
y = df_all_years['bankrupt'].copy()

# Check for missing values
print(f"\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
missing_counts = X.isnull().sum()
missing_pct = (missing_counts / len(X) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print("\nFeatures with missing values:")
print(missing_df[missing_df['Missing Count'] > 0].head(15))

# For baseline model, drop rows with any missing values in features
print(f"\nRows before dropping missing: {len(X):,}")
complete_mask = X.notna().all(axis=1)
X_complete = X[complete_mask]
y_complete = y[complete_mask]
print(f"Rows after dropping missing: {len(X_complete):,}")
print(f"Percentage retained: {len(X_complete)/len(X)*100:.1f}%")
print(f"Bankruptcy rate in complete data: {y_complete.mean():.2%}")

# Split data - stratified to maintain class balance
print(f"\n" + "="*60)
print("TRAIN/TEST SPLIT")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X_complete, y_complete,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_complete
)

print(f"Train set: {len(X_train):,} samples, {y_train.mean():.2%} bankrupt")
print(f"Test set: {len(X_test):,} samples, {y_test.mean():.2%} bankrupt")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression with balanced class weights
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*60)

model = LogisticRegression(
    class_weight='balanced',  # Handle class imbalance
    random_state=RANDOM_STATE,
    max_iter=1000,
    solver='lbfgs'
)

print("Fitting model...")
model.fit(X_train_scaled, y_train)
print("Model training complete")

# Cross-validation on training set
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=CV_FOLDS,
    scoring='roc_auc'
)
print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Performance metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("\nTraining Set Performance:")
print("-" * 40)
print(classification_report(y_train, y_train_pred, target_names=['Non-Bankrupt', 'Bankrupt']))
train_roc_auc = roc_auc_score(y_train, y_train_proba)
print(f"ROC-AUC: {train_roc_auc:.4f}")

print("\nTest Set Performance:")
print("-" * 40)
print(classification_report(y_test, y_test_pred, target_names=['Non-Bankrupt', 'Bankrupt']))
test_roc_auc = roc_auc_score(y_test, y_test_proba)
print(f"ROC-AUC: {test_roc_auc:.4f}")

# Confusion matrices
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives: {cm[1,1]:,}")

# Feature importance (coefficients)
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Top 20)")
print("="*60)

feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output_dir = script_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Save feature importance
feature_importance.to_csv(output_dir / 'all_years_feature_importance.csv', index=False)
print(f"Feature importance saved to: all_years_feature_importance.csv")

# Save model performance metrics
results = {
    'model': 'Logistic Regression (All Years)',
    'date': datetime.now().isoformat(),
    'data': {
        'total_observations': len(df_all_years),
        'complete_cases': len(X_complete),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'bankruptcy_rate': float(y_complete.mean()),
        'years_used': [2016, 2017, 2018]
    },
    'features': {
        'total_features': len(available_features),
        'feature_list': available_features
    },
    'performance': {
        'cross_validation': {
            'mean_roc_auc': float(cv_scores.mean()),
            'std_roc_auc': float(cv_scores.std()),
            'cv_folds': CV_FOLDS
        },
        'train': {
            'roc_auc': float(train_roc_auc)
        },
        'test': {
            'roc_auc': float(test_roc_auc),
            'confusion_matrix': cm.tolist()
        }
    }
}

with open(output_dir / 'all_years_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Model results saved to: all_years_results.json")

# Save detailed predictions for analysis
predictions_df = pd.DataFrame({
    'Orgnr': df_all_years.loc[X_complete.index, 'Orgnr'],
    'year': df_all_years.loc[X_complete.index, 'year'],
    'actual_bankrupt': y_complete,
    'predicted_bankrupt': np.concatenate([y_train_pred, y_test_pred]),
    'probability': np.concatenate([y_train_proba, y_test_proba]),
    'dataset': ['train'] * len(X_train) + ['test'] * len(X_test)
})
predictions_df.to_csv(output_dir / 'all_years_predictions.csv', index=False)
print(f"Predictions saved to: all_years_predictions.csv")

print("\n" + "="*60)
print("ALL YEARS MODEL COMPLETE")
print("="*60)
print(f"\nKey Difference from 2018_only model:")
print(f"- Used {len(X_complete):,} observations across 3 years")
print(f"- 2018_only model used ~52K observations from 1 year")
print(f"- This model has {len(X_complete)/52303:.1f}x more data points")
