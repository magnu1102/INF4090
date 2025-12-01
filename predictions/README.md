# Bankruptcy Prediction Models

This folder contains machine learning models for predicting company bankruptcy in Norway using 2018 data to predict 2019 bankruptcy status.

## Theoretical Foundation

Our approach is grounded in seminal bankruptcy prediction research:

- **Beaver (1966)**: Working capital ratios and cash flow measures
- **Altman (1968)**: Z-Score multivariate discriminant analysis
- **Ohlson (1980)**: Logistic regression approach to bankruptcy prediction

## Data Overview

- **Source**: `data/features/feature_dataset_v1.parquet`
- **Total 2018 companies**: 90,138
- **Complete cases used**: 52,303 (58% of 2018 data)
- **Bankruptcy rate**: 2.07% (1,083 bankrupt companies)
- **Train/Test split**: 80/20 stratified split

## Baseline Model: Logistic Regression

### Model Configuration

- **Algorithm**: Logistic Regression with L2 regularization
- **Class weighting**: Balanced (to handle 2% bankruptcy rate)
- **Solver**: LBFGS
- **Feature scaling**: StandardScaler (mean=0, std=1)
- **Random state**: 42 (for reproducibility)

### Feature Selection

The baseline model uses 24 features across five categories:

1. **Financial Ratios** (7 features)
   - likviditetsgrad_1, likviditetsgrad_2
   - total_gjeldsgrad, egenkapitalandel
   - rentedekningsgrad, driftsmargin
   - totalkapitalrentabilitet

2. **Altman Z-Score** (1 feature)
   - altman_z_score

3. **Temporal Features** (4 features)
   - omsetningsvekst_1617, omsetningsvekst_1718
   - fallende_likviditet, konsistent_underskudd

4. **Missingness Indicators** (3 features)
   - levert_alle_år, levert_2018, regnskapskomplett

5. **Company Characteristics** (3 features)
   - selskapsalder, nytt_selskap, log_totalkapital

6. **Warning Signals** (3 features)
   - negativ_egenkapital, sterkt_overbelånt, lav_likviditet

7. **Auditor Changes** (3 features)
   - byttet_revisor_1617, byttet_revisor_1718, byttet_revisor_noensinne

### Performance Metrics

#### Cross-Validation (5-fold)
- **Mean ROC-AUC**: 0.9696 ± 0.0179
- Consistent performance across folds (range: 0.9549 - 0.9798)

#### Test Set Performance
- **ROC-AUC**: 0.9726
- **Accuracy**: 95%

#### Classification Report (Test Set)
```
              precision    recall  f1-score   support
Non-Bankrupt       1.00      0.95      0.97     10,244
    Bankrupt       0.27      0.88      0.41        217
```

#### Confusion Matrix (Test Set)
```
                  Predicted
                  Non-B  Bankrupt
Actual Non-B      9,733    511
Actual Bankrupt      27    190
```

#### Key Performance Indicators
- **True Positive Rate (Recall)**: 88% - Model catches 88% of bankruptcies
- **False Negative Rate**: 12% - Model misses 12% of bankruptcies
- **Precision**: 27% - When model predicts bankruptcy, it's correct 27% of the time
- **False Positive Rate**: 5% - 5% of healthy companies flagged as at-risk

### Model Interpretation

#### Top 10 Most Important Features

1. **levert_alle_år** (coef: -1.85) - Companies that filed all years are much less likely to go bankrupt
2. **altman_z_score** (coef: -0.59) - Higher Z-scores indicate financial health
3. **log_totalkapital** (coef: -0.44) - Larger companies are less likely to fail
4. **negativ_egenkapital** (coef: +0.42) - Negative equity strongly predicts bankruptcy
5. **lav_likviditet** (coef: +0.29) - Low liquidity is a warning signal
6. **sterkt_overbelånt** (coef: +0.26) - High leverage increases risk
7. **nytt_selskap** (coef: +0.22) - Younger companies are at higher risk
8. **byttet_revisor_noensinne** (coef: -0.17) - Auditor changes associated with distress
9. **fallende_likviditet** (coef: +0.16) - Declining liquidity predicts failure
10. **byttet_revisor_1718** (coef: +0.15) - Recent auditor changes are concerning

### Key Insights

1. **Non-filing is highly predictive**: The single most important predictor is whether a company filed all required financial statements (`levert_alle_år`). This aligns with prior findings that 76.5% of bankrupt companies didn't file 2018 data.

2. **Altman Z-Score effectiveness**: The multivariate Z-Score is the second most important feature, validating Altman's (1968) approach for Norwegian data.

3. **Company size matters**: Larger companies (`log_totalkapital`) have significantly lower bankruptcy risk, supporting the "too big to fail" hypothesis.

4. **Balance sheet health**: Traditional warning signs (negative equity, high leverage, low liquidity) remain strong predictors.

5. **Temporal dynamics**: Year-over-year changes (falling liquidity, auditor changes) add predictive value beyond static ratios.

### Model Trade-offs

The baseline model prioritizes **high recall (88%)** at the expense of **low precision (27%)**. This means:

- **Strengths**: Catches most bankruptcies (only 12% missed)
- **Weakness**: Many false alarms (73% of bankruptcy predictions are incorrect)

For bankruptcy prediction, high recall is generally preferred because:
- Missing a bankruptcy is costly (credit losses, unpaid invoices)
- False alarms can be filtered through human review
- The model serves as an early warning system, not a final decision

### Files Generated

- `baseline_model.py` - Training script
- `baseline_results.json` - Machine-readable performance metrics
- `baseline_feature_importance.csv` - Complete feature ranking
- `baseline_predictions.csv` - Per-company predictions for analysis
- `README.md` - This documentation

## Next Steps

1. **Advanced Models**: Compare with Random Forest, XGBoost, Neural Networks
2. **Feature Engineering**: Test polynomial features, interactions
3. **Threshold Optimization**: Adjust decision threshold for different recall/precision trade-offs
4. **SMOTE**: Test synthetic minority oversampling
5. **Temporal Models**: Use panel data structure for time-series predictions
6. **Ensemble Methods**: Combine multiple model predictions

## Usage

To retrain the baseline model:

```bash
cd INF4090/predictions
python baseline_model.py
```

The script will:
1. Load feature dataset
2. Filter to 2018 data
3. Handle missing values (drop incomplete rows)
4. Split into train/test sets
5. Train logistic regression
6. Generate performance metrics
7. Save results and predictions
