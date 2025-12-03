"""
Sector C bankruptcy risk modeling script

Usage:
    python helperMethods/sector_C_bankruptcy_model.py \
        --input data/features/feature_dataset_v1.csv \
        --outdir outputs/sector_C_model

This script builds a supervised classifier for companies in NACE Section C (Manufacturing).
It performs:
 - filtering to Section C using `Næringskode`
 - preprocessing (impute, scale)
 - stratified cross-validation with LogisticRegression and XGBoost baselines
 - class-imbalance handling (class_weight)
 - model evaluation (ROC AUC, PR AUC, Brier score, confusion matrix)
 - feature importance and SHAP explanations

The script creates next-year bankruptcy labels to avoid leakage.
"""
import os
import sys
import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, 
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    xgb = None
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

warnings.filterwarnings('ignore')


NACE_SECTION_RANGES = {
    'A': (1, 3),
    'B': (5, 9),
    'C': (10, 33),
    'D': (35, 35),
    'E': (36, 39),
    'F': (41, 43),
    'G': (45, 47),
    'H': (49, 53),
    'I': (55, 56),
    'J': (58, 63),
    'K': (64, 66),
    'L': (68, 68),
    'M': (69, 75),
    'N': (77, 82),
    'O': (84, 84),
    'P': (85, 85),
    'Q': (86, 88),
    'R': (90, 93),
    'S': (94, 96),
    'T': (97, 98),
    'U': (99, 99),
}


def nace_to_section(code):
    """Map a NACE numeric code to section letter. Handles formats like '10.11', '1011', '10'."""
    if pd.isna(code):
        return None
    s = str(code).strip()
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) == 0:
        return None
    try:
        val = int(digits[:2])
    except ValueError:
        return None
    for sec, (lo, hi) in NACE_SECTION_RANGES.items():
        if lo <= val <= hi:
            return sec
    return None


def load_and_filter(input_csv, sector_letter='C', nrows=None):
    """Load CSV in chunks and filter to sector."""
    print(f"Loading data from {input_csv} and filtering to sector {sector_letter}...")
    
    # Try to find NACE column
    df_sample = pd.read_csv(input_csv, nrows=1)
    nace_col = None
    for col in df_sample.columns:
        if 'næringskode' in col.lower() or 'naeringskode' in col.lower() or 'nace' in col.lower():
            nace_col = col
            break
    
    if nace_col is None:
        raise RuntimeError('No NACE/Næringskode column found in dataset')
    
    print(f"Using column '{nace_col}' for sector mapping")
    
    # Read in chunks
    reader = pd.read_csv(input_csv, dtype=str, chunksize=200000)
    parts = []
    total_read = 0
    
    for chunk in reader:
        chunk['section'] = chunk[nace_col].apply(nace_to_section)
        sel = chunk['section'] == sector_letter
        if sel.any():
            parts.append(chunk.loc[sel])
        total_read += len(chunk)
        if total_read % 500000 == 0:
            print(f"  Processed {total_read} rows...")
        if nrows is not None and sum(len(p) for p in parts) >= nrows:
            break
    
    if len(parts) == 0:
        raise RuntimeError(f'No rows found for sector {sector_letter}')
    
    df = pd.concat(parts, ignore_index=True)
    if nrows is not None:
        df = df.head(nrows)
    
    print(f"Loaded {len(df)} rows for sector {sector_letter}")
    return df


def prepare_labels_and_features(df, label_col='bankrupt', year_col='year', orgnr_col='Orgnr'):
    """Create next-year bankruptcy label and extract numeric features."""
    print(f"Preparing labels and features...")
    
    # Normalize label
    if label_col not in df.columns:
        raise RuntimeError(f'Label column {label_col} not found')
    
    y_raw = df[label_col].astype(str).str.strip().str.upper()
    y_raw = y_raw.replace({'J':'1', 'JA':'1', 'Y':'1', 'YES':'1', '1':'1', 
                           'N':'0', 'NEI':'0', '0':'0', 'FALSE':'0', 'TRUE':'1'})
    df['bankrupt_flag'] = y_raw.map({'1':1, '0':0}).fillna(-1).astype(int)
    
    # Convert year to numeric
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df = df[df[year_col].notna()].copy()
    df[year_col] = df[year_col].astype(int)
    
    # Create next-year label (prevent leakage)
    df = df.sort_values([orgnr_col, year_col])
    df['bankrupt_next'] = df.groupby(orgnr_col)['bankrupt_flag'].shift(-1)
    
    # Keep only rows with valid next-year label
    valid = (df['bankrupt_next'] >= 0) & (df['bankrupt_flag'] >= 0)
    df = df[valid].copy()
    df['bankrupt_next'] = df['bankrupt_next'].astype(int)
    
    print(f"Created {len(df)} training examples with next-year labels")
    print(f"  Positive rate: {df['bankrupt_next'].mean():.3f}")
    
    # Identify numeric features
    numeric_cols = []
    for col in df.columns:
        if col in (label_col, year_col, orgnr_col, 'Navn', 'Forretningsadresse', 
                   'Beskrivelse til næringskode', 'Beskrivelse til sektorkode', 'section', 'bankrupt_flag', 'bankrupt_next'):
            continue
        try:
            tmp = pd.to_numeric(df[col], errors='coerce')
            if tmp.notna().sum() >= 10:
                numeric_cols.append(col)
        except Exception:
            continue
    
    print(f"Selected {len(numeric_cols)} numeric features")
    
    X = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Replace infinities with NaN, then imputation will handle them
    X = X.replace([np.inf, -np.inf], np.nan)
    
    y = df['bankrupt_next'].values
    
    return X, y, numeric_cols, df[orgnr_col].values


def evaluate_model(y_true, y_pred_proba, model_name='model'):
    """Compute evaluation metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'model': model_name,
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier': float(brier),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    
    return metrics


def run_cv_pipeline(X, y, outdir, model_type='logistic'):
    """Run stratified cross-validation and return out-of-fold predictions."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    if model_type == 'logistic':
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    elif model_type == 'xgb' and HAS_XGB:
        scale_pos_weight = (len(y) - y.sum()) / (y.sum() + 1e-5)
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=4
        )
    
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('clf', clf),
    ])
    
    print(f"Running 5-fold stratified CV with {model_type}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get out-of-fold predictions
    probs = cross_val_predict(
        pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1
    )
    pos_probs = probs[:, 1]
    
    metrics = evaluate_model(y, pos_probs, model_name=model_type)
    
    # Save metrics
    with open(outdir / f'metrics_{model_type}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR AUC:  {metrics['pr_auc']:.4f}")
    print(f"  Brier:   {metrics['brier']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # Train final model on all data for feature importance
    pipeline.fit(X, y)
    
    # Save feature importances
    if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
        fi = pipeline.named_steps['clf'].feature_importances_
        fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=False)
        fi_series.to_csv(outdir / f'feature_importances_{model_type}.csv')
        
        print(f"\nTop 10 features ({model_type}):")
        for i, (feat, score) in enumerate(fi_series.head(10).items(), 1):
            print(f"  {i}. {feat}: {score:.4f}")
    
    # SHAP for XGBoost
    if model_type == 'xgb' and HAS_SHAP and HAS_XGB:
        try:
            print("Computing SHAP values...")
            xgb_model = pipeline.named_steps['clf']
            explainer = shap.TreeExplainer(xgb_model)
            X_sample = X.sample(min(1000, len(X)), random_state=42)
            shap_vals = explainer.shap_values(X_sample)
            
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
            shap_importance.to_csv(outdir / 'shap_importance_xgb.csv')
            
            print("SHAP importance (sample):")
            for i, (feat, score) in enumerate(shap_importance.head(10).items(), 1):
                print(f"  {i}. {feat}: {score:.4f}")
        except Exception as e:
            print(f"  SHAP computation failed: {e}")
    
    return metrics, pos_probs


def main():
    parser = argparse.ArgumentParser(
        description='Train bankruptcy prediction model for Sector C (Manufacturing)'
    )
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--outdir', default='outputs/sector_C_model', help='Output directory')
    parser.add_argument('--sector', default='C', help='NACE sector (default: C)')
    parser.add_argument('--label', default='bankrupt', help='Bankruptcy label column')
    parser.add_argument('--nrows', type=int, default=None, help='Max rows to process')
    
    args = parser.parse_args()
    
    # Load and filter data
    df = load_and_filter(args.input, sector_letter=args.sector, nrows=args.nrows)
    
    # Prepare features and labels
    X, y, numeric_cols, groups = prepare_labels_and_features(df, label_col=args.label)
    
    if len(X) == 0:
        raise RuntimeError('No valid data after preprocessing')
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run models
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION (Baseline)")
    print("="*60)
    metrics_log, probs_log = run_cv_pipeline(X, y, outdir, model_type='logistic')
    
    print("\n" + "="*60)
    print("RANDOM FOREST")
    print("="*60)
    metrics_rf, probs_rf = run_cv_pipeline(X, y, outdir, model_type='rf')
    
    if HAS_XGB:
        print("\n" + "="*60)
        print("XGBOOST")
        print("="*60)
        metrics_xgb, probs_xgb = run_cv_pipeline(X, y, outdir, model_type='xgb')
    else:
        metrics_xgb = None
        print("\nXGBoost not installed, skipping...")
    
    # Summary report
    summary = {
        'dataset': {
            'sector': args.sector,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'positive_rate': float(y.mean()),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum())
        },
        'models': [metrics_log, metrics_rf]
    }
    if metrics_xgb:
        summary['models'].append(metrics_xgb)
    
    with open(outdir / 'summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SAVED TO:", outdir)
    print("="*60)
    print(f"Summary: {outdir / 'summary_report.json'}")
    for model in ['logistic', 'rf', 'xgb']:
        metrics_file = outdir / f'metrics_{model}.json'
        if metrics_file.exists():
            print(f"Metrics:  {metrics_file}")


if __name__ == '__main__':
    main()
