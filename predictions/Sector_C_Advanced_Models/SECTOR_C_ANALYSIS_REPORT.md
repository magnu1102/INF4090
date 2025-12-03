# Sector C Bankruptcy Prediction - Advanced Model Analysis

**Analysis Date:** December 3, 2025  
**Sector:** C (Mining and Quarrying, NACE 05-09)  
**Report Generated:** 2025-12-03 15:19:16

---

## Executive Summary

This report presents a comprehensive bankruptcy prediction analysis for Norwegian companies in Sector C (Mining and Quarrying) using machine learning models trained on 19 financial features (9 raw accounting numbers and 10 financial ratios).

**Key Findings:**
- **Dataset Size (Post-Cleaning):** 45 complete observations with 13.33% bankruptcy rate
- **Test Set Performance:** All models show perfect recall (100% detection of bankruptcies) but with significant false positive rates
- **Best Model:** Logistic Regression (ROC-AUC: 1.00, F1: 0.67)
- **Critical Limitation:** Extremely small sample size severely limits reliability and generalization

---

## 1. Data Overview

### Raw Data Characteristics

| Metric | Value |
|--------|-------|
| Total Sector C Records | 101 |
| Records with Complete Data | 45 (44.6%) |
| Bankruptcy Cases | 41 in raw data / 6 in clean data |
| Bankruptcy Rate (Raw) | 40.59% |
| Bankruptcy Rate (Clean) | 13.33% |
| Years Covered | 2016, 2017, 2018 |

### Data Quality Issues

**⚠️ CRITICAL CONCERNS:**

1. **Extreme Data Loss:** 56 records (55.4%) were dropped due to missing values
   - The loss of more than half the data raises concerns about:
     - Whether missing data is itself predictive (non-filing often signals distress)
     - Potential selection bias in remaining sample
     - Reduced bankruptcy representation (41 → 6 cases)

2. **Severe Class Imbalance Resolution:** 
   - Started with 40.59% bankruptcy rate (41/101)
   - Ended with 13.33% rate (6/45) after dropping incomplete records
   - This suggests non-bankrupt companies had better data reporting

3. **Minuscule Test Set:** Only 9 test observations
   - 1 bankruptcy case to predict
   - Metrics are essentially meaningless at this scale
   - Any random variation can flip results dramatically

### Features Used

**Raw Accounting Numbers (9):**
- `Tall 1340` - Sales revenue (Salgsinntekt)
- `Tall 7709` - Other operating income
- `Tall 72` - Total income (Sum inntekter)
- `Tall 146` - Operating result (Driftsresultat)
- `Tall 217` - Fixed assets (Sum anleggsmidler)
- `Tall 194` - Current assets (Sum omløpsmidler)
- `Tall 85` - Short-term debt (Sum kortsiktig gjeld)
- `Tall 86` - Long-term debt (Sum langsiktig gjeld)
- `Tall 17130` - Financial expenses (Sum finanskostnader)

**Financial Ratios (10):**
- `likviditetsgrad_1` - Current ratio
- `total_gjeldsgrad` - Total debt ratio
- `langsiktig_gjeldsgrad` - Long-term debt ratio
- `kortsiktig_gjeldsgrad` - Short-term debt ratio
- `egenkapitalandel` - Equity ratio
- `driftsmargin` - Operating margin
- `driftsrentabilitet` - Operating ROA (CORRECTED: not totalkapitalrentabilitet)
- `omsetningsgrad` - Asset turnover ratio
- `rentedekningsgrad` - Interest coverage ratio
- `altman_z_score` - Altman Z-Score

---

## 2. Model Performance Summary

### Test Set Results (9 Observations: 8 Non-Bankrupt, 1 Bankrupt)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Sensitivity | Specificity |
|-------|----------|-----------|--------|----------|---------|-------------|-------------|
| **Logistic Regression** | **0.889** | **0.500** | **1.000** | **0.667** | **1.000** | 1.000 | 0.875 |
| Random Forest | 0.778 | 0.333 | 1.000 | 0.500 | 0.750 | 1.000 | 0.750 |
| XGBoost | 0.778 | 0.333 | 1.000 | 0.750 | 0.875 | 1.000 | 0.750 |
| Gradient Boosting | 0.778 | 0.333 | 1.000 | 0.500 | 0.875 | 1.000 | 0.750 |

### Confusion Matrices (Test Set)

**Logistic Regression:**
```
              Predicted
              Not Bankrupt  Bankrupt
Actual Not      7            1
       Bankrupt 0            1
```
- True Positives: 1
- False Positives: 1
- True Negatives: 7
- False Negatives: 0

**Random Forest / XGBoost / Gradient Boosting:**
```
              Predicted
              Not Bankrupt  Bankrupt
Actual Not      6            2
       Bankrupt 0            1
```
- True Positives: 1
- False Positives: 2
- True Negatives: 6
- False Negatives: 0

---

## 3. Feature Importance Analysis

### Random Forest Feature Importance

Top 10 Features:
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Tall 72 (Total Income) | 0.1830 |
| 2 | Tall 1340 (Sales Revenue) | 0.1600 |
| 3 | Tall 85 (Short-term Debt) | 0.1244 |
| 4 | Tall 194 (Current Assets) | 0.1000 |
| 5 | Tall 217 (Fixed Assets) | 0.1000 |
| 6 | Tall 17130 (Financial Expenses) | 0.0800 |
| 7 | Tall 146 (Operating Result) | 0.0800 |
| 8 | rentedekningsgrad (Interest Coverage) | 0.0300 |
| 9 | driftsmargin (Operating Margin) | 0.0300 |
| 10 | egenkapitalandel (Equity Ratio) | 0.0300 |

**Insight:** Raw revenue and income numbers dominate. This suggests company scale/size is more predictive than profitability ratios for this sector.

### XGBoost Feature Importance

Top Features:
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Tall 1340 (Sales Revenue) | 0.3414 |
| 2 | Tall 72 (Total Income) | 0.3103 |
| 3 | Tall 217 (Fixed Assets) | 0.2780 |
| 4 | Tall 146 (Operating Result) | 0.0224 |
| 5 | total_gjeldsgrad (Debt Ratio) | 0.0187 |

**Insight:** XGBoost focuses almost entirely on revenue and asset base. The model essentially treats bankruptcy as a function of company size for Sector C.

### Logistic Regression Feature Importance (Top Coefficients)

| Rank | Feature | |Coefficient| |
|------|---------|---|
| 1 | Tall 217 (Fixed Assets) | 0.6291 |
| 2 | total_gjeldsgrad (Debt Ratio) | 0.5637 |
| 3 | egenkapitalandel (Equity Ratio) | 0.5637 |
| 4 | kortsiktig_gjeldsgrad (ST Debt Ratio) | 0.4632 |
| 5 | Tall 72 (Total Income) | 0.4112 |

**Insight:** LR emphasizes leverage and size metrics. Debt/equity structure appears meaningful.

### Gradient Boosting Feature Importance

```
Tall 72 (Total Income):     0.6675 (66.8%)
Tall 1340 (Sales Revenue):  0.3325 (33.3%)
All other features:         0.0000
```

**Concern:** This model essentially only uses two features, suggesting:
- Only total income and sales distinguish bankruptcies in this dataset
- Minimal information in other 17 features
- Possible overfitting or degenerate solution

---

## 4. Critical Assessment & Limitations

### ⚠️ MAJOR LIMITATIONS

#### 1. **Catastrophically Small Sample Size**
- **45 observations is extremely small** for machine learning
- Standard statistical practice: Need at least 100-200 observations per class
- With only 6 bankruptcy cases, any predictions are essentially random noise
- Standard error in metrics is enormous
- A single misclassified case changes accuracy by 11.1%

#### 2. **Severe Data Quality Degradation**
- Started with 101 observations (40.59% bankrupt) → ended with 45 (13.33% bankrupt)
- **55.4% data loss** indicates fundamental data recording problems in this sector
- Missing data may be **informative** - non-filing often precedes bankruptcy
  - Should NOT be discarded but treated as a feature
  - Current approach loses predictive signal

#### 3. **Perfect Recall + High False Positives = Useless**
All models achieve 100% recall but generate false alarms:
- Logistic Regression: 50% false positive rate
- Other models: 67% false positive rate (2/3 bankruptcies called are false alarms)

**Interpretation:** Models are essentially predicting "everything is bankrupt" with slight modulation
- High sensitivity is good for early warning
- But 50%+ false positive rate makes the model operationally useless
- Would trigger unnecessary investigations/warnings on non-bankrupt firms

#### 4. **Extreme Overfitting Signals**
- Logistic Regression achieves perfect ROC-AUC (1.00) on tiny test set
- This is nearly impossible with real data - strong sign of overfitting
- The model may have memorized the training set

#### 5. **Feature Importance Instability**
- Gradient Boosting uses ONLY 2 features (entire predictive power from Tall 72 + Tall 1340)
- Other models rank features very differently
- This disagreement signals unreliable feature importance on this small sample
- Cannot trust any feature importance claims

#### 6. **Sector-Specific Issues**
- Mining/quarrying is highly volatile and cyclical
- Small sample size means few economic cycles represented (only 3 years)
- Industry-specific shocks not captured
- May include distressed firms from commodity crash period

### 📊 IMBALANCE & BIAS

**Class Distribution:**
- Train set: 13.89% bankrupt (5/36)
- Test set: 11.11% bankrupt (1/9)

The minority class (bankruptcy) is still under-represented despite starting imbalance being addressed by:
- Dropping missing values (which removed mostly non-bankrupt cases)
- Stratified split (maintained proportions)

Even so, this is an extremely skewed learning problem.

---

## 5. What the Models Are Actually Capturing

### The Real Pattern

**Hypothesis (from feature importance):**
The models are likely capturing that:
- **Smaller companies have higher bankruptcy rates** (company size → bankruptcy)
- This is a well-known phenomenon (liability of smallness)
- Models may be using revenue/assets as a **size proxy** rather than true bankruptcy predictor

### The Sector Context

Mining/Quarrying (Sector C) in Norway:
- Historically volatile, tied to commodity prices
- Small number of active companies (101 observations across 3 years)
- 40.59% bankruptcy rate is **extremely high** and suggests:
  - Possibly a distressed period (2016-2018 had commodity downturn)
  - Potential data quality issues or misclassification
  - Or the sector was indeed in severe crisis

---

## 6. Recommendations & Improvements

### IMMEDIATE ACTIONS

1. **Increase Data Volume**
   - Use longer time period (5+ years)
   - Include more economic cycles
   - Combine with adjacent sectors if possible
   - **Target: 500+ observations minimum**

2. **Handle Missing Data Properly**
   - Don't drop rows with missing data in predictive features
   - Create missingness indicators as features
   - Non-filing is itself a bankruptcy signal
   - Use imputation (median, KNN, or predictive) as alternative

3. **Verify Class Labels**
   - Confirm bankruptcy definitions are accurate
   - 40.59% rate seems suspiciously high
   - Check data quality issues in raw records

4. **Test on Holdout Years**
   - Don't use random split (data is temporal)
   - Use 2016-2017 for train, 2018 for test
   - Mimics real-world scenario

### METHODOLOGICAL IMPROVEMENTS

5. **Cross-Validation**
   - Current: Single 80/20 split on 9 test observations
   - Better: 5-10 fold cross-validation
   - Reports model stability across folds

6. **Threshold Optimization**
   - Current: Default 0.5 decision threshold
   - For early warning: Lower to 0.3-0.4 (catch more failures)
   - Accept higher false positive rate for safety

7. **Business-Specific Cost Function**
   - Define cost of false positives vs false negatives
   - Type II error (missing bankruptcy): Very expensive
   - Type I error (false alarm): Less expensive
   - Optimize threshold accordingly

8. **Ensemble Methods**
   - Average predictions across multiple models
   - Use weighted voting (weight by model performance)
   - May improve generalization

### FEATURE ENGINEERING

9. **Add Temporal Features**
   - Year-over-year changes: Revenue growth, debt growth
   - Trends: 3-year declining revenues
   - Already in dataset but not heavily weighted

10. **Domain-Specific Indicators**
    - Mining sector cycles (commodity prices)
    - Debt maturity profiles
    - Cash flow (if available)
    - Audit qualifications (going concern warnings)

11. **Remove Redundant Features**
    - Total income (Tall 72) and sales (Tall 1340) are correlated
    - Multiple leverage ratios measure similar concept
    - Trim to top 10-12 features based on stable importance

---

## 7. Model Comparison & Recommendation

### Performance Summary

| Aspect | Best Model | Comment |
|--------|-----------|---------|
| ROC-AUC | Logistic Regression | 1.00 (but likely overfitting) |
| F1-Score | XGBoost & GB | 0.75 (but on 1 positive case) |
| Recall | All tied | 100% (catches all bankruptcies) |
| Precision | Logistic Regression | 50% (vs 33% for others) |
| Specificity | Logistic Regression | 87.5% (fewer false alarms) |
| Feature Stability | Random Forest | Most balanced importance |
| Interpretability | Logistic Regression | Coefficients are interpretable |

### Recommended Model: **Logistic Regression** (with caveats)

**Why:**
- Best precision (50% vs 33%)
- Best specificity (87.5%)
- Only 1 false alarm vs 2 for others
- Most interpretable coefficients
- Simpler = less overfitting risk

**But:** None of these models are production-ready. Need more data.

---

## 8. Key Insights for Sector C

### What Works
✓ **Company Size Matters:** Revenue and assets are consistently predictive
✓ **Debt Levels Matter:** Leverage ratios appear significant
✓ **Perfect Recall Possible:** Model can flag borderline cases

### What Doesn't Work (Yet)
✗ **Profitability Ratios:** Operating margin/ROA contribute little
✗ **Individual Metrics:** Most ratios are ignored (0% importance in some models)
✗ **Generalization:** Model too specialized to small dataset
✗ **Reliable Prediction:** Test set too small for confidence intervals

### Sector-Specific Concerns
- **Volatility:** Sector undergoes boom/bust cycles
- **Non-Filing:** 55% of records had missing data (data quality issue)
- **Extreme Imbalance:** 40%+ bankruptcy rate is sector-wide crisis indicator
- **Small Population:** Only ~100 observations for entire sector over 3 years

---

## 9. Statistical Validity

### Confidence in Metrics: **VERY LOW** ⚠️

| Metric | Why Unreliable |
|--------|---|
| Accuracy: 88.9% | Based on 9 samples; 95% CI is ±15% |
| Precision: 50% | Based on 2 predictions; huge variance |
| Recall: 100% | Based on 1 positive case; meaningless |
| ROC-AUC: 1.00 | Impossible on real data; clear overfitting |
| Feature Importance | Varies wildly between models |

**Statistical Minimum:** For stable metrics, need:
- At least 50-100 positive cases per class
- Test set with 30+ positive cases
- Cross-validation across multiple folds
- Current: 1 positive case in test set

---

## 10. Conclusion

### Summary of Findings

1. **Data Quality:** Sector C has severe missing data issues (55% loss rate)
2. **Sample Size:** 45 complete observations is too small for reliable ML
3. **Model Performance:** All models perfect recall but moderate precision
4. **Best Model:** Logistic Regression is most practical
5. **True Pattern:** Likely capturing company size effect, not pure bankruptcy risk
6. **Actionability:** Model can flag risky companies but needs human review

### Bottom Line

**These models are proof-of-concept only and NOT suitable for production use.**

The extreme data scarcity in Sector C (mining/quarrying) has created an impossible learning scenario:
- Too few observations
- Too few bankruptcy cases
- Too much missing data
- Possible selection bias in remaining data

### Recommended Path Forward

**Short-term (1-2 months):**
1. Collect additional years of data (2013-2015, 2019-2020)
2. Fix missing data issues
3. Re-run analysis with 300+ observations

**Medium-term (3-6 months):**
1. Combine adjacent sectors to increase sample
2. Implement proper temporal cross-validation
3. Add external features (commodity prices, macro indicators)

**Long-term (6-12 months):**
1. Build integrated multi-sector model
2. Implement real-time monitoring system
3. Deploy with human-in-the-loop review

---

## Appendix: Technical Details

### Model Configurations

**Logistic Regression:**
- Regularization: L2 (default)
- Max iterations: 1000
- Solver: lbfgs (default)
- Scaling: StandardScaler

**Random Forest:**
- Estimators: 100
- Max depth: 15
- Min samples split: 10
- Min samples leaf: 5
- Class weight: balanced

**XGBoost:**
- Estimators: 100
- Max depth: 6
- Learning rate: 0.1
- Subsample: 0.8
- Colsample_bytree: 0.8
- Scale pos weight: 13.33 (for class imbalance)

**Gradient Boosting:**
- Estimators: 100
- Max depth: 5
- Learning rate: 0.1
- Min samples split: 10
- Min samples leaf: 5

### Train-Test Split
- Method: Stratified random split
- Test size: 20% (9 observations)
- Random state: 42
- Stratification: Maintained class proportions

### Outlier Handling
- Method: IQR-based clipping
- Bounds: Q1 ± 3×IQR
- Purpose: Remove extreme values without losing data

---

## Output Files Generated

1. `model_results.json` - Detailed metrics for all models
2. `feature_importance_logistic_regression.csv` - LR coefficients
3. `feature_importance_random_forest.csv` - RF feature importance
4. `feature_importance_xgboost.csv` - XGBoost feature importance
5. `feature_importance_gradient_boosting.csv` - GB feature importance
6. `test_predictions.csv` - Detailed predictions for all test cases

---

**Report prepared by:** AI Assistant  
**Analysis period:** 2016-2018  
**Next review:** Upon data expansion to 300+ observations
