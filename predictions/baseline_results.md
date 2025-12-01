# Baseline Bankruptcy Prediction Model - Results

**Model:** Logistic Regression with Balanced Class Weights
**Date:** 2025-12-01
**Prediction Task:** Binary classification - predict 2019 bankruptcy using 2018 data

---

## Executive Summary

The baseline logistic regression model achieved **excellent performance** with:
- **Test ROC-AUC: 0.9726** (97.26% area under ROC curve)
- **Cross-validation ROC-AUC: 0.9696 ± 0.0179** (5-fold CV)
- **88% Recall** on bankrupt companies (detected 190 out of 217 bankruptcies)
- **95% Precision** on non-bankrupt companies

The model demonstrates strong predictive power, with the top predictor being **levert_alle_år** (filed all years), followed by **altman_z_score** and **log_totalkapital** (company size).

---

## Data Preparation

### Dataset Overview
- **Total 2018 companies:** 90,138
- **Complete cases (no missing values):** 52,303 (58.0%)
- **Bankruptcy rate:** 2.07% (class imbalance present)
- **Train/Test split:** 80/20 stratified split
  - Training set: 41,842 samples (2.07% bankrupt)
  - Test set: 10,461 samples (2.07% bankrupt)

### Features Used
**24 features** selected based on theoretical foundation (Beaver 1966, Altman 1968, Ohlson 1980):

#### Financial Ratios (8 features)
- `likviditetsgrad_1` - Current ratio (Beaver)
- `likviditetsgrad_2` - Quick ratio
- `total_gjeldsgrad` - Total debt ratio
- `egenkapitalandel` - Equity ratio
- `rentedekningsgrad` - Interest coverage ratio
- `driftsmargin` - Operating margin
- `totalkapitalrentabilitet` - Return on total assets
- `altman_z_score` - Altman's bankruptcy prediction score

#### Temporal Features (4 features)
- `omsetningsvekst_1617` - Revenue growth 2016-2017
- `omsetningsvekst_1718` - Revenue growth 2017-2018
- `fallende_likviditet` - Declining liquidity indicator
- `konsistent_underskudd` - Consistent losses indicator

#### Missingness Indicators (2 features)
- `levert_alle_år` - Filed financial statements for all years
- `levert_2018` - Filed 2018 financial statements
- `regnskapskomplett` - Complete accounting data

#### Company Characteristics (4 features)
- `selskapsalder` - Company age (years)
- `nytt_selskap` - New company indicator
- `log_totalkapital` - Log of total assets (company size)

#### Warning Signals (3 features)
- `negativ_egenkapital` - Negative equity indicator
- `sterkt_overbelånt` - Highly leveraged indicator
- `lav_likviditet` - Low liquidity indicator

#### Auditor Changes (3 features)
- `byttet_revisor_1617` - Changed auditor 2016-2017
- `byttet_revisor_1718` - Changed auditor 2017-2018
- `byttet_revisor_noensinne` - Ever changed auditor

### Missing Data Handling
- **Strategy:** Complete case analysis (listwise deletion)
- **Rows with missing values:** 37,835 (42.0%) excluded from analysis
- **Rationale:** Establish clean baseline; advanced imputation can be explored in future models

---

## Model Performance

### Test Set Results

#### Classification Metrics
```
              precision    recall  f1-score   support

Non-Bankrupt       1.00      0.95      0.97     10,244
    Bankrupt       0.27      0.88      0.41        217

    accuracy                           0.95     10,461
```

#### Key Performance Indicators
- **ROC-AUC:** 0.9726 (excellent discrimination)
- **Accuracy:** 95.0%
- **Precision (Bankrupt):** 27% (511 false positives)
- **Recall (Bankrupt):** 88% (only 27 false negatives)

#### Confusion Matrix
```
                 Predicted
                 Non-Bank  Bankrupt
Actual Non-Bank     9,733       511
Actual Bankrupt        27       190
```

- **True Negatives:** 9,733 (correctly identified non-bankrupt)
- **False Positives:** 511 (non-bankrupt flagged as bankrupt)
- **False Negatives:** 27 (bankrupt missed by model)
- **True Positives:** 190 (correctly identified bankrupt)

### Cross-Validation Results
- **5-Fold CV ROC-AUC:** 0.9696 ± 0.0179
- **Fold scores:** [0.9777, 0.9701, 0.9657, 0.9549, 0.9798]
- **Interpretation:** Model performance is stable and consistent across folds

### Training Set Performance
- **ROC-AUC:** 0.9749 (slightly higher than test, minimal overfitting)
- **Recall (Bankrupt):** 88%
- **Precision (Bankrupt):** 28%

---

## Feature Importance

### Top 10 Most Important Features

The logistic regression coefficients indicate feature importance. **Negative coefficients** reduce bankruptcy probability; **positive coefficients** increase it.

| Rank | Feature | Coefficient | Interpretation |
|------|---------|-------------|----------------|
| 1 | `levert_alle_år` | -1.850 | Filing all years **strongly reduces** bankruptcy risk |
| 2 | `altman_z_score` | -0.587 | Higher Z-score **reduces** bankruptcy risk (Altman 1968) |
| 3 | `log_totalkapital` | -0.440 | Larger companies have **lower** bankruptcy risk |
| 4 | `negativ_egenkapital` | +0.416 | Negative equity **strongly increases** bankruptcy risk |
| 5 | `lav_likviditet` | +0.295 | Low liquidity **increases** bankruptcy risk |
| 6 | `sterkt_overbelånt` | +0.262 | High leverage **increases** bankruptcy risk |
| 7 | `nytt_selskap` | +0.225 | New companies have **higher** bankruptcy risk |
| 8 | `byttet_revisor_noensinne` | -0.166 | Ever changing auditor **reduces** risk (unexpected) |
| 9 | `fallende_likviditet` | +0.161 | Declining liquidity **increases** bankruptcy risk |
| 10 | `byttet_revisor_1718` | +0.145 | Recent auditor change **increases** bankruptcy risk |

### Key Insights

1. **Non-filing is the strongest predictor:** Companies that don't file financial statements for all years have dramatically higher bankruptcy risk (coefficient: -1.85)

2. **Traditional bankruptcy theory validated:** Altman Z-Score (-0.587), negative equity (+0.416), and low liquidity (+0.295) are all strong predictors, consistent with Beaver (1966) and Altman (1968)

3. **Company size matters:** Larger companies (log_totalkapital: -0.440) have lower bankruptcy risk

4. **Auditor changes are complex:**
   - Recent changes (2017-2018) **increase** bankruptcy risk (+0.145)
   - But ever changing auditor **decreases** risk (-0.166)
   - This suggests survivor bias or that distressed companies avoid changing auditors

5. **Temporal trends are informative:** Declining liquidity (+0.161) and consistent losses (+0.105) predict bankruptcy

---

## Model Interpretation

### Strengths
1. **Excellent discrimination:** ROC-AUC of 0.9726 indicates the model can distinguish bankrupt from non-bankrupt companies
2. **High recall (88%):** The model catches most bankruptcies, critical for early warning systems
3. **Theoretically grounded:** Features align with established bankruptcy prediction literature
4. **Stable performance:** Low variance in cross-validation scores
5. **Interpretable:** Linear model with clear coefficient interpretation

### Limitations
1. **Low precision (27%):** High false positive rate means many non-bankrupt companies flagged as at-risk
2. **Class imbalance:** Only 2.07% bankruptcy rate; used balanced class weights to compensate
3. **Complete case analysis:** 42% of data excluded due to missing values
4. **Linear assumptions:** Logistic regression assumes linear relationships between features and log-odds

### Trade-offs
The model is tuned with **balanced class weights**, prioritizing **recall over precision**:
- **High recall (88%):** Only 27 bankruptcies missed - good for early warning
- **Low precision (27%):** 511 false alarms - acceptable if intervention costs are low
- **Use case dependent:** For screening, high recall is desirable; for final decisions, may need higher precision

---

## Comparison to Theory

### Beaver (1966) - Working Capital Ratios
- **Validated:** Current ratio (`likviditetsgrad_1`, coef: -0.119) and cash flow measures are predictive
- **Key finding:** Liquidity measures perform as expected

### Altman (1968) - Z-Score Model
- **Validated:** Altman Z-Score is the **2nd most important** feature (coef: -0.587)
- **Key finding:** Multivariate approach remains highly effective 56 years later

### Ohlson (1980) - Logistic Regression
- **Validated:** Logistic regression approach achieves ROC-AUC of 0.9726
- **Key finding:** Ohlson's logit model framework is appropriate for Norwegian data

---

## Business Implications

### For Creditors and Lenders
- Model can identify 88% of future bankruptcies one year in advance
- Consider using probability thresholds to balance false positives/negatives
- Non-filing behavior is the strongest red flag

### For Regulators
- Companies with negative equity and declining liquidity require monitoring
- Small, new companies have elevated bankruptcy risk
- Auditor changes in final year before bankruptcy are a warning signal

### For Investors
- Company size (log_totalkapital) and filing compliance are protective factors
- Altman Z-Score remains a valuable screening tool
- Consider multiple financial ratios rather than single metrics

---

## Next Steps

### Potential Model Improvements
1. **Handle missing data:** Implement imputation or use algorithms that handle missingness (e.g., XGBoost)
2. **Address class imbalance:** Test SMOTE, undersampling, or ensemble methods
3. **Non-linear models:** Try Random Forest, Gradient Boosting, or Neural Networks
4. **Feature engineering:** Create interaction terms, polynomial features
5. **Threshold optimization:** Tune decision threshold based on business costs

### Advanced Analysis
1. **Feature selection:** Test stepwise selection, LASSO regularization
2. **Model comparison:** Benchmark against Random Forest, XGBoost, Neural Networks
3. **Temporal validation:** Use 2016 data to predict 2017, 2017→2018 as robustness checks
4. **Industry analysis:** Build industry-specific models
5. **Ensemble methods:** Combine multiple models for improved performance

### Research Questions
1. How do feature importances differ across ML algorithms? (Original research question)
2. Can we improve precision without sacrificing recall?
3. What is the optimal prediction horizon (1 year, 2 years, 3 years)?
4. Do results generalize to post-2019 data?

---

## Files Generated

1. **baseline_feature_importance.csv** - Full list of features ranked by absolute coefficient
2. **baseline_results.json** - Machine-readable performance metrics
3. **baseline_predictions.csv** - Individual company predictions with probabilities
4. **baseline_results.md** - This comprehensive report

---

## Conclusion

The baseline logistic regression model demonstrates that **bankruptcy prediction is feasible** with 2018 data, achieving a test ROC-AUC of **0.9726**. The model validates classical bankruptcy prediction theory (Beaver, Altman, Ohlson) while revealing that **non-filing behavior** is the single strongest predictor.

The high recall (88%) makes this model suitable for **screening and early warning**, though the low precision (27%) suggests it should be used as a first-stage filter rather than a final decision tool. Future work should explore more sophisticated algorithms and techniques to improve precision while maintaining recall.

This baseline establishes a strong foundation for comparing feature importance across different machine learning algorithms, directly addressing the thesis research question.
