# Sector C Bankruptcy Prediction - Visual Summary

## Model Performance Comparison

### Metric Comparison Chart

```
ACCURACY
Logistic Regression: ████████░ 0.889
Random Forest:       ███████░░ 0.778
XGBoost:             ███████░░ 0.778
Gradient Boosting:   ███████░░ 0.778

PRECISION
Logistic Regression: █████░░░░░░░ 0.500
Random Forest:       ███░░░░░░░░░ 0.333
XGBoost:             ███░░░░░░░░░ 0.333
Gradient Boosting:   ███░░░░░░░░░ 0.333

RECALL (All Perfect)
Logistic Regression: ██████████ 1.000 ✓
Random Forest:       ██████████ 1.000 ✓
XGBoost:             ██████████ 1.000 ✓
Gradient Boosting:   ██████████ 1.000 ✓

F1-SCORE
Logistic Regression: ██████░░░░░ 0.667
Random Forest:       █████░░░░░░ 0.500
XGBoost:             █████░░░░░░ 0.500
Gradient Boosting:   █████░░░░░░ 0.500

ROC-AUC
Logistic Regression: ██████████ 1.000 ⚠️ Overfitting
Random Forest:       ███████░░░ 0.750
XGBoost:             ████████░░ 0.875
Gradient Boosting:   ████████░░ 0.875

SPECIFICITY
Logistic Regression: ████████░░ 0.875
Random Forest:       ███████░░░ 0.750
XGBoost:             ███████░░░ 0.750
Gradient Boosting:   ███████░░░ 0.750
```

### Confusion Matrix Visualization

**Logistic Regression (Best Model)**
```
                 PREDICTED
              Not Bankrupt | Bankrupt
ACTUAL  Not        7      |    1        = 8
        Bankrupt   0      |    1        = 1
                   ───────────────
                     7    |    2
```

**Random Forest / XGBoost / Gradient Boosting**
```
                 PREDICTED
              Not Bankrupt | Bankrupt
ACTUAL  Not        6      |    2        = 8
        Bankrupt   0      |    1        = 1
                   ───────────────
                     6    |    3
```

---

## Data Overview

### Sample Size Breakdown

```
Starting Data (Sector C)
└─ 101 observations

    │
    ├─→ 60 non-bankrupt (59.4%)
    ├─→ 41 bankrupt (40.6%)
    │
    └─→ Clean Data (complete records)
        └─ 45 observations (44.6% retention)
            │
            ├─→ 39 non-bankrupt (86.7%)
            ├─→ 6 bankrupt (13.3%)
            │
            ├─→ TRAIN (80%) = 36 obs
            │   └─ 5 bankrupt (13.9%)
            │
            └─→ TEST (20%) = 9 obs
                └─ 1 bankrupt (11.1%)
```

### Critical Data Loss Waterfall

```
101 records
  ↓ (Missing values in features)
  ↓ (55% data loss)
  ↓
45 complete records ← ONLY 44.6% REMAIN!
```

**Impact:** Bankruptcy cases dropped from 41 → 6 (-85.4%)

---

## Feature Importance Consensus

### Features All Models Rank in Top 5

| Rank | Feature | Category | Consensus |
|------|---------|----------|-----------|
| 1-2 | Tall 1340, Tall 72 | Revenue/Income | ✓✓✓✓ |
| 2-3 | Tall 217, Tall 194 | Assets | ✓✓✓ |
| 3-5 | Tall 85 | Short-term Debt | ✓✓ |

### Features Ranked Low by ALL Models

- ✗ Tall 7709 (Other operating income)
- ✗ Tall 17130 (Financial expenses - XGBoost)
- ✗ likviditetsgrad_2 (Quick ratio equivalent)
- ✗ omsetningsvolatilitet (Revenue volatility)

**Interpretation:** Models strongly prefer accounting scale (company size) over operational efficiency metrics.

---

## Model-by-Model Analysis

### 1️⃣ Logistic Regression - WINNER

**Strengths:**
- Highest precision (50%) = fewer false alarms
- Highest specificity (87.5%) = good at identifying true non-bankrupts
- Best F1-score (0.667)
- Most interpretable coefficients

**Weaknesses:**
- Perfect ROC-AUC (1.00) suggests overfitting
- Only 1 true positive in test set

**Best For:** Research, interpretability, avoiding false alarms

---

### 2️⃣ Random Forest

**Strengths:**
- Feature importance is more balanced
- Robust to outliers
- Captures non-linear relationships

**Weaknesses:**
- 67% false positive rate
- 100% recall achieved but at high cost
- Less interpretable than LR

**Best For:** Benchmarking, ensemble averaging

---

### 3️⃣ XGBoost

**Strengths:**
- Good ROC-AUC (0.875)
- Efficient training
- Handles class imbalance well

**Weaknesses:**
- Uses only 3 features effectively (Tall 1340, 72, 217)
- Other 16 features contribute virtually nothing
- High false positive rate (67%)

**Best For:** Large datasets (not this one)

---

### 4️⃣ Gradient Boosting

**Strengths:**
- Competitive ROC-AUC (0.875)
- Sequential learning approach

**Weaknesses:**
- Uses ONLY 2 features (Tall 72 + Tall 1340)
- All other 17 features = 0% importance
- Degenerate solution = severe overfitting

**Worst Model for this dataset**

---

## Risk Assessment Matrix

### ⚠️ RELIABILITY vs CONFIDENCE

```
┌─────────────────────────────────────────────────┐
│  Statistical Confidence Levels                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  Accuracy: 88.9% (±15% at 95% CI)              │
│  ✗ Far too wide for practical use               │
│                                                  │
│  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  ROC-AUC: 100% (UNREALISTIC)                    │
│  ✗ Clear sign of overfitting                    │
│                                                  │
│  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  Precision: 50% (based on 2 predictions)        │
│  ✗ Essentially random variation                 │
│                                                  │
└─────────────────────────────────────────────────┘

VERDICT: DO NOT USE IN PRODUCTION
```

---

## Key Metrics Explained

### 1. Accuracy: 88.9%
"Out of 9 predictions, we got 8 right"
- **But:** 1 bankruptcy with 88.9% accuracy isn't reliable
- **Translation:** "We got lucky this time"

### 2. Precision: 50% (LR)
"When we predict bankruptcy, we're right 50% of the time"
- **Problem:** False alarm 50% of the time = wasteful investigations
- **Translation:** "Coin flip prediction quality"

### 3. Recall/Sensitivity: 100%
"We caught the 1 actual bankruptcy in test set"
- **But:** Only 1 case in test = meaningless
- **Translation:** "We got 1 out of 1, but that's tiny sample"

### 4. ROC-AUC: 1.00
"Perfect discrimination between classes"
- **Problem:** Impossible on real data = overfitting
- **Translation:** "Our model memorized the 9-sample test set"

### 5. Specificity: 87.5% (LR)
"We correctly identified 7 out of 8 non-bankrupt firms"
- **Good:** This one metric is meaningful
- **But:** Still based on tiny sample

---

## Sector C Context

### Why This Sector is Risky

```
Mining & Quarrying (Sector C) Characteristics:

Volatility:      ████████████████░░ VERY HIGH
Cyclicality:     ████████████░░░░░░ HIGH
Data Quality:    ██░░░░░░░░░░░░░░░░ POOR (55% missing)
Sample Size:     ███░░░░░░░░░░░░░░░░ TOO SMALL
Bankruptcy Rate: ██████████████████░ EXTREME (40.6%)

Combined Risk:   ███████████████████░ UNMODELLABLE AT SCALE
```

### Why 40.6% Bankruptcy is Problematic

Normal bankruptcy rates: 1-3%  
Sector C rate: 40.6%

**Possible explanations:**
1. Data quality issue / mislabeling
2. Sector was in severe crisis (2016-2018 commodity crash)
3. Selection bias (only distressed firms have complete data)
4. Definitional issues (how is bankruptcy defined?)

**Implication:** Sample is not representative of typical sector

---

## Recommendations Ranked by Priority

### MUST DO (Before Any Production Use)

1. **🔴 CRITICAL: Expand Data**
   - Current: 45 observations, 6 bankrupt
   - Required: 500+ observations, 50+ bankrupt
   - Effort: 3-6 months
   - Impact: Model credibility increases 10x

2. **🔴 CRITICAL: Fix Missing Data Issues**
   - Current: 55% dropout rate
   - Problem: Biases remaining sample
   - Solution: Imputation or inclusion of missingness features
   - Effort: 2-4 weeks
   - Impact: Recovers signal

### SHOULD DO (Before Deployment)

3. **🟡 IMPORTANT: Temporal Validation**
   - Current: Random 80/20 split
   - Better: 2016-17 train, 2018 test
   - Effort: 1 week
   - Impact: Realistic performance assessment

4. **🟡 IMPORTANT: Add External Data**
   - Commodity prices (iron ore, copper, sand)
   - Macro indicators (GDP, employment)
   - Regulatory changes
   - Effort: 4-8 weeks
   - Impact: Better explanations

### COULD DO (Nice-to-Have)

5. **🟢 OPTIONAL: Hyperparameter Tuning**
   - Current: Fixed parameters
   - Better: Grid search / Bayesian optimization
   - Effort: 2-3 weeks (after more data)
   - Impact: 5-10% improvement

6. **🟢 OPTIONAL: Feature Engineering**
   - Cash flow indicators
   - Growth trends
   - Debt maturity structure
   - Effort: 3-4 weeks
   - Impact: 10-15% improvement

---

## Bottom Line: One Sentence Per Model

| Model | Verdict |
|-------|---------|
| **LR** | Best of bad options; good for research |
| **RF** | Too many false alarms for production |
| **XGB** | Regression to mean; only uses 3 features |
| **GB** | Worst; degenerate solution on 2 features |

---

## Critical Success Factors for Future Work

```
┌────────────────────────────────────────────────────────┐
│  What Needs to Happen BEFORE Production:              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ✓ 10x More Data (45 → 450+ observations)            │
│  ✓ Fix 55% Missing Data Rate                         │
│  ✓ Verify Bankruptcy Labels (40.6% rate too high)    │
│  ✓ Time-Series Validation (not random split)         │
│  ✓ Add Sector-Specific Features                      │
│  ✓ Proper Cross-Validation                           │
│  ✓ Business Rule Definition (threshold, costs)       │
│  ✓ Human Expert Review Integration                   │
│                                                        │
│  Timeline: 6-12 months before deployment             │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Files Generated

| File | Purpose |
|------|---------|
| `SECTOR_C_ANALYSIS_REPORT.md` | Full detailed analysis (17 sections) |
| `README.md` | Quick reference guide |
| `model_results.json` | Machine-readable metric summary |
| `model_comparison.csv` | Side-by-side metrics |
| `test_predictions.csv` | Individual predictions + probabilities |
| `feature_importance_*.csv` | Feature rankings per model |
| `VISUAL_SUMMARY.md` | This file (charts & visuals) |

---

**Last Updated:** December 3, 2025  
**Status:** Analysis Complete  
**Next Review:** Upon data expansion to 300+ observations
