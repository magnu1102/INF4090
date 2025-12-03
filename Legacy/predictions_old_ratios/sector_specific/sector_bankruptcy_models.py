"""
Sector-Specific Bankruptcy Prediction Models
=============================================

This script builds separate logistic regression models for specific næringskoder:
- C: Industri (Manufacturing)
- F: Byggje- og anleggsverksemd (Construction)
- G: Varehandel, reparasjon av motorvogner (Retail/Motor vehicle repair)
- I: Overnattings- og serveringsverksemd (Accommodation and food service)

For each sector, we compare bankrupt vs non-bankrupt companies using:
- All years (2016, 2017, 2018)
- Same features as all_years model
- Supervised logistic regression with balanced class weights
- Complete case analysis (listwise deletion)

Goal: Identify sector-specific bankruptcy predictors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SECTOR-SPECIFIC BANKRUPTCY PREDICTION MODELS")
print("="*70)

RANDOM_STATE = 42
CV_FOLDS = 5

# Define sectors to analyze (using NACE code ranges)
SECTORS = {
    'C': {'name': 'Industri', 'range': (10, 33)},
    'F': {'name': 'Byggje- og anleggsverksemd', 'range': (41, 43)},
    'G': {'name': 'Varehandel, reparasjon av motorvogner', 'range': (45, 47)},
    'I': {'name': 'Overnattings- og serveringsverksemd', 'range': (55, 56)}
}

# Load data
script_dir = Path(__file__).parent
input_file = script_dir.parent.parent / 'data' / 'features' / 'feature_dataset_v1.parquet'

print(f"\nLoading data from: {input_file}")
df = pd.read_parquet(input_file)

# Use all years (2016, 2017, 2018)
df_all = df[df['year'].isin([2016, 2017, 2018])].copy()
print(f"\nTotal observations (all sectors): {len(df_all):,}")
print(f"Overall bankruptcy rate: {df_all['bankrupt'].mean():.2%}")

# Extract numeric NACE code from Næringskode (first 2 digits)
def extract_nace_code(naringskode):
    """Extract first 2 digits of NACE code as integer"""
    try:
        code_str = str(naringskode).split('.')[0]  # Get part before decimal
        if code_str and code_str.strip() and code_str.strip()[0].isdigit():
            return int(code_str[:2]) if len(code_str) >= 2 else int(code_str[0]) if code_str[0].isdigit() else None
        return None
    except:
        return None

df_all['nace_code'] = df_all['Næringskode'].apply(extract_nace_code)

print("\n" + "="*70)
print("SECTOR DISTRIBUTION")
print("="*70)

sector_summary = []
for sector_code, sector_info in SECTORS.items():
    sector_name = sector_info['name']
    min_code, max_code = sector_info['range']

    # Filter by NACE code range
    sector_df = df_all[(df_all['nace_code'] >= min_code) & (df_all['nace_code'] <= max_code)]
    n_companies = sector_df['Orgnr'].nunique()
    n_obs = len(sector_df)
    n_bankrupt = sector_df['bankrupt'].sum()
    bankr_rate = sector_df['bankrupt'].mean()

    sector_summary.append({
        'Sector_Code': sector_code,
        'Sector_Name': sector_name,
        'Unique_Companies': n_companies,
        'Total_Observations': n_obs,
        'Bankruptcies': n_bankrupt,
        'Bankruptcy_Rate': bankr_rate
    })

    print(f"\n{sector_code}: {sector_name}")
    print(f"  Unique companies: {n_companies:,}")
    print(f"  Total observations: {n_obs:,}")
    print(f"  Bankruptcies: {n_bankrupt:,}")
    print(f"  Bankruptcy rate: {bankr_rate:.2%}")

# ============================================================================
# FEATURE SELECTION (Same as all_years model)
# ============================================================================

print("\n" + "="*70)
print("FEATURE SELECTION")
print("="*70)

# Financial ratios (engineered features)
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

# Growth features
growth_features = [
    'omsetningsvekst_1617',
    'omsetningsvekst_1718',
    'aktivavekst_1617',
    'aktivavekst_1718',
    'gjeldsvekst_1617',
    'gjeldsvekst_1718',
    'omsetningsvolatilitet',
]

# Warning signals
warning_signals = [
    'negativ_egenkapital',
    'sterkt_overbelånt',
    'lav_likviditet',
    'driftsunderskudd',
    'fallende_likviditet',
    'konsistent_underskudd',
    'økende_gjeldsgrad',
]

# Company characteristics
company_chars = [
    'selskapsalder',
    'nytt_selskap',
    'log_totalkapital',
    'log_omsetning',
    'Antall ansatte',
]

# Filing behavior (highly predictive)
filing_features = [
    'levert_alle_år',
    'levert_2018',
    'antall_år_levert',
    'regnskapskomplett',
]

# Combine all features
feature_columns = (
    financial_ratios +
    growth_features +
    warning_signals +
    company_chars +
    filing_features
)

# Filter to available features
feature_columns = [f for f in feature_columns if f in df_all.columns]

print(f"\nTotal features to use: {len(feature_columns)}")
print(f"  Financial ratios: {len([f for f in financial_ratios if f in df_all.columns])}")
print(f"  Growth features: {len([f for f in growth_features if f in df_all.columns])}")
print(f"  Warning signals: {len([f for f in warning_signals if f in df_all.columns])}")
print(f"  Company characteristics: {len([f for f in company_chars if f in df_all.columns])}")
print(f"  Filing behavior: {len([f for f in filing_features if f in df_all.columns])}")

# ============================================================================
# BUILD MODELS FOR EACH SECTOR
# ============================================================================

all_sector_results = {}

for sector_code, sector_info in SECTORS.items():
    sector_name = sector_info['name']
    min_code, max_code = sector_info['range']

    print("\n" + "="*70)
    print(f"SECTOR {sector_code}: {sector_name}")
    print(f"NACE codes: {min_code}-{max_code}")
    print("="*70)

    # Filter to this sector
    sector_df = df_all[(df_all['nace_code'] >= min_code) & (df_all['nace_code'] <= max_code)].copy()

    print(f"\nObservations in sector: {len(sector_df):,}")
    print(f"Bankruptcy rate: {sector_df['bankrupt'].mean():.2%}")

    # Prepare features and target
    X = sector_df[feature_columns].copy()
    y = sector_df['bankrupt'].copy()

    print(f"\nFeature matrix shape: {X.shape}")

    # Convert all columns to numeric (some like 'Antall ansatte' may be stored as strings)
    print("\nConverting features to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Handle infinity values
    print("Handling infinity values...")
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    # Complete case analysis (listwise deletion)
    missing_before = X.isnull().any(axis=1).sum()
    complete_cases = ~X.isnull().any(axis=1)
    X_complete = X[complete_cases].copy()
    y_complete = y[complete_cases].copy()

    missing_after = len(sector_df) - len(X_complete)
    print(f"\nMissing data handling (complete case analysis):")
    print(f"  Observations with missing data: {missing_before:,} ({missing_before/len(sector_df)*100:.1f}%)")
    print(f"  Complete cases: {len(X_complete):,} ({len(X_complete)/len(sector_df)*100:.1f}%)")
    print(f"  Bankruptcy rate in complete cases: {y_complete.mean():.2%}")

    # Check if we have enough data
    if len(X_complete) < 100:
        print(f"\n⚠️  WARNING: Only {len(X_complete)} complete cases - too small for reliable modeling")
        all_sector_results[sector_code] = {
            'sector_name': sector_name,
            'error': 'Insufficient data',
            'n_observations': len(X_complete)
        }
        continue

    if y_complete.sum() < 10:
        print(f"\n⚠️  WARNING: Only {y_complete.sum()} bankruptcies - too few events for modeling")
        all_sector_results[sector_code] = {
            'sector_name': sector_name,
            'error': 'Insufficient bankruptcy events',
            'n_bankruptcies': int(y_complete.sum())
        }
        continue

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_complete)

    # Train logistic regression with balanced class weights
    print("\nTraining logistic regression (balanced class weights)...")
    model = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver='lbfgs'
    )

    # Cross-validation
    print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    cv_results = cross_validate(
        model, X_scaled, y_complete,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    print("\nCross-validation results:")
    print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} (+/- {cv_results['test_roc_auc'].std():.4f})")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
    print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")

    # Train final model on all data
    print("\nTraining final model on all complete cases...")
    model.fit(X_scaled, y_complete)

    # Get predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    roc_auc = roc_auc_score(y_complete, y_pred_proba)
    precision = precision_score(y_complete, y_pred)
    recall = recall_score(y_complete, y_pred)
    f1 = f1_score(y_complete, y_pred)

    print("\nFinal model performance (on training data):")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_complete, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:,}")
    print(f"  False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}")
    print(f"  True Positives:  {cm[1,1]:,}")

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\nTop 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        direction = "+ Bankruptcy" if row['Coefficient'] > 0 else "- Bankruptcy"
        print(f"  {row['Feature']:40s} {row['Coefficient']:+.4f} ({direction})")

    # Store results
    all_sector_results[sector_code] = {
        'sector_name': sector_name,
        'n_total_observations': len(sector_df),
        'n_complete_cases': len(X_complete),
        'n_bankruptcies': int(y_complete.sum()),
        'bankruptcy_rate': float(y_complete.mean()),
        'cv_scores': {
            'roc_auc_mean': float(cv_results['test_roc_auc'].mean()),
            'roc_auc_std': float(cv_results['test_roc_auc'].std()),
            'precision_mean': float(cv_results['test_precision'].mean()),
            'precision_std': float(cv_results['test_precision'].std()),
            'recall_mean': float(cv_results['test_recall'].mean()),
            'recall_std': float(cv_results['test_recall'].std()),
            'f1_mean': float(cv_results['test_f1'].mean()),
            'f1_std': float(cv_results['test_f1'].std())
        },
        'final_metrics': {
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'confusion_matrix': {
            'TN': int(cm[0,0]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0]),
            'TP': int(cm[1,1])
        },
        'top_features': feature_importance.head(15).to_dict('records')
    }

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = script_dir

# Save detailed results JSON
results_json = {
    'analysis_date': datetime.now().isoformat(),
    'description': 'Sector-specific bankruptcy prediction models',
    'sectors_analyzed': list(SECTORS.keys()),
    'features_used': feature_columns,
    'n_features': len(feature_columns),
    'sectors': all_sector_results
}

json_path = output_dir / 'sector_bankruptcy_results.json'
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Detailed results saved to: {json_path}")

# Save sector summary CSV
summary_rows = []
for sector_code, results in all_sector_results.items():
    if 'error' in results:
        summary_rows.append({
            'Sector_Code': sector_code,
            'Sector_Name': results['sector_name'],
            'Status': 'Error: ' + results['error'],
            'N_Observations': results.get('n_observations', 0),
            'N_Bankruptcies': results.get('n_bankruptcies', 0)
        })
    else:
        summary_rows.append({
            'Sector_Code': sector_code,
            'Sector_Name': results['sector_name'],
            'Status': 'Success',
            'N_Total_Observations': results['n_total_observations'],
            'N_Complete_Cases': results['n_complete_cases'],
            'N_Bankruptcies': results['n_bankruptcies'],
            'Bankruptcy_Rate': f"{results['bankruptcy_rate']:.2%}",
            'CV_ROC_AUC': f"{results['cv_scores']['roc_auc_mean']:.4f}",
            'CV_Precision': f"{results['cv_scores']['precision_mean']:.4f}",
            'CV_Recall': f"{results['cv_scores']['recall_mean']:.4f}",
            'Final_ROC_AUC': f"{results['final_metrics']['roc_auc']:.4f}",
            'Final_Precision': f"{results['final_metrics']['precision']:.4f}",
            'Final_Recall': f"{results['final_metrics']['recall']:.4f}"
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = output_dir / 'sector_bankruptcy_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

# Save feature importance for each sector
for sector_code, results in all_sector_results.items():
    if 'top_features' in results:
        features_df = pd.DataFrame(results['top_features'])
        features_path = output_dir / f'sector_{sector_code}_feature_importance.csv'
        features_df.to_csv(features_path, index=False)
        print(f"Feature importance for sector {sector_code} saved to: {features_path}")

print("\n" + "="*70)
print("SECTOR-SPECIFIC ANALYSIS COMPLETE")
print("="*70)

print("\nSummary:")
for sector_code, sector_info in SECTORS.items():
    sector_name = sector_info['name']
    if sector_code in all_sector_results:
        results = all_sector_results[sector_code]
        if 'error' in results:
            print(f"\n{sector_code} ({sector_name}): ⚠️  {results['error']}")
        else:
            print(f"\n{sector_code} ({sector_name}):")
            print(f"  Complete cases: {results['n_complete_cases']:,}")
            print(f"  Bankruptcies: {results['n_bankruptcies']:,} ({results['bankruptcy_rate']:.2%})")
            print(f"  CV ROC-AUC: {results['cv_scores']['roc_auc_mean']:.4f}")
