# Sector C Bankruptcy Prediction - Model Results

**Status:** Analysis Complete  
**Date:** December 3, 2025  
**Sector:** C (Mining & Quarrying, NACE 05-09)

## Quick Summary

Four machine learning models were trained to predict bankruptcy for Norwegian companies in Sector C:
1. **Logistic Regression** - Best performer (ROC-AUC: 1.00, F1: 0.67)
2. **Random Forest** - Balanced approach (ROC-AUC: 0.75, F1: 0.50)
3. **XGBoost** - Competitive (ROC-AUC: 0.875, F1: 0.50)
4. **Gradient Boosting** - Ensemble (ROC-AUC: 0.875, F1: 0.50)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Sector C Records | 101 |
| Complete Records | 45 (44.6%) |
| Bankruptcy Cases | 6 (13.33%) |
| Train Set | 36 (13.89% bankrupt) |
| Test Set | 9 (11.11% bankrupt) |

## ⚠️ CRITICAL LIMITATION

**Sample size of 45 observations is TOO SMALL for reliable machine learning predictions.**

Standard ML practice requires 500+ observations with 50+ cases per class. This analysis should be considered:
- **Proof of concept** only
- **NOT suitable for production**
- **Requires data expansion** before deployment

## File Structure

```
Sector_C_Advanced_Models/
├── SECTOR_C_ANALYSIS_REPORT.md          # Full analysis with detailed critique
├── model_results.json                    # Metric summary (all 4 models)
├── test_predictions.csv                  # Individual predictions for test set
├── feature_importance_logistic_regression.csv
├── feature_importance_random_forest.csv
├── feature_importance_xgboost.csv
└── feature_importance_gradient_boosting.csv
```

## Key Findings

### Model Performance (Test Set)

```
Logistic Regression:
  Accuracy:    88.89%  (8/9 correct)
  Precision:   50.00%  (1/2 bankruptcies predicted were correct)
  Recall:      100.00% (caught 1/1 actual bankruptcy)
  F1-Score:    66.67%
  ROC-AUC:     1.0000  ⚠️ Likely overfitting
  
Random Forest / XGBoost / Gradient Boosting:
  Accuracy:    77.78%  (7/9 correct)
  Precision:   33.33%  (1/3 bankruptcies predicted were correct)
  Recall:      100.00% (caught 1/1 actual bankruptcy)
  F1-Score:    50.00%
  ROC-AUC:     0.75-0.875
```

### Top Predictive Features

**Rank 1-3 (all models agree):**
1. Tall 72 (Total Income / Sum inntekter)
2. Tall 1340 (Sales Revenue / Salgsinntekt)
3. Company Scale / Asset Base

**Pattern:** Models strongly emphasize company size over traditional bankruptcy indicators.

### Major Limitations

1. **Tiny test set:** Only 9 observations, 1 bankruptcy
2. **High data loss:** 55% of original data had missing values
3. **Perfect recall + false alarms:** 100% catch rate but 50-67% false positive rate
4. **Overfitting signals:** Perfect ROC-AUC is impossible on real data
5. **Extreme class imbalance in surviving sample:** Went from 40.59% → 13.33% bankruptcy rate

## Recommendations

### BEFORE Using These Models

**DO NOT deploy to production.** Instead:

1. **Expand data:** Collect 2013-2015 and 2019-2020 observations
2. **Fix missing data:** 55% loss rate indicates fundamental issues
3. **Verify labels:** 40.59% bankruptcy rate seems unusually high
4. **Temporal validation:** Use proper time-series split (train on 2016-17, test on 2018)

### For Sector C Specifically

- **Volatility matters:** Mining is cyclical; need multiple economic cycles in training
- **Non-filing is a signal:** Don't discard missing data; treat as feature
- **Add external data:** Commodity prices, macro indicators, regulatory changes
- **Consider survival bias:** Only observed companies can go bankrupt; selection bias in data

## Technical Details

- **Training approach:** 80/20 stratified random split
- **Features:** 9 raw accounting values + 10 financial ratios (19 total)
- **Scaling:** StandardScaler for Logistic Regression, raw values for tree models
- **Outlier handling:** IQR-based clipping (±3 IQR)
- **Class imbalance:** Addressed via stratification and balanced class weights
- **Hyperparameter tuning:** Fixed parameters (no grid search due to small sample)

## Next Steps

1. Read `SECTOR_C_ANALYSIS_REPORT.md` for detailed critique and insights
2. Review feature importance CSVs to understand model drivers
3. Examine `test_predictions.csv` to see individual prediction probabilities
4. Plan data collection to expand sample to 300+ observations
5. Implement temporal cross-validation once more data available

## Questions & Caveats

**Q: Why is ROC-AUC = 1.0?**  
A: On a 9-sample test set with 1 positive case, perfect separation is possible due to randomness. This almost certainly indicates overfitting rather than actual predictive power.

**Q: Why such high false positive rate?**  
A: Models are conservative (call things bankruptcy) to achieve 100% recall. 50-67% false alarms make this operationally risky.

**Q: Should we use these in production?**  
A: **Absolutely not.** Use as internal research only. Need 10x more data first.

**Q: Why did 55% of data get dropped?**  
A: Missing values in financial ratios require complete data for most fields. The original dataset has data quality issues, especially in Sector C.

---

**Analysis performed:** December 3, 2025 15:19  
**Report generated:** By AI Assistant (GitHub Copilot)  
**For:** Sector C bankruptcy prediction research
