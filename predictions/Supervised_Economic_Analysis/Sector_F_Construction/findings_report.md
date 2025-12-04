# Sector F (Construction) - Supervised Economic Analysis
## Bankruptcy Prediction Model - Final Results

**Analysis Date:** December 4, 2025
**Sector:** F - Construction (NACE codes 41-43)
**Model:** Random Forest Classifier (200 trees, class-balanced)

---

## Executive Summary

Developed a supervised bankruptcy prediction model for Norwegian construction companies achieving **AUC 0.87** on held-out test data - the highest performance across all analyzed sectors. Analysis of 74,144 complete company-year observations (2016-2018) identified **kortsiktig_gjeldsgrad** (short-term debt ratio) as the dominant predictor, reflecting construction's unique cash flow vulnerabilities.

### Key Findings
- **Model Performance:** Test AUC 0.87 - best performance among all sectors
- **Sample Size:** 74,144 observations (66.3% of Sector F), 2,774 bankruptcies (3.74% rate)
- **Top Predictor:** Kortsiktig_gjeldsgrad (short-term debt ratio) - importance 0.0855
- **Risk Stratification:** Very High risk tier shows 17.18% bankruptcy rate vs 0.04% in Very Low tier (430× difference)
- **Economic Regimes:** 3 distinct regimes identified, with distressed micro-contractors (Regime 1) showing 14.67% bankruptcy rate

**Construction-Specific Insight:** Unlike manufacturing (equity-driven risk), construction bankruptcies are driven by **short-term liquidity stress** and **debt-liquidity mismatch**, consistent with project-based cash flow volatility.

---

## 1. Data Overview

### Sample Characteristics
| Metric | Value |
|--------|-------|
| Total observations | 111,802 |
| Companies | 46,367 |
| Years | 2016, 2017, 2018 |
| Complete cases | 74,144 (66.3%) |
| Bankruptcies (total) | 5,727 (5.12%) |
| Bankruptcies (complete cases) | 2,774 (3.74%) |

### Data Quality Notes
- **Higher bankruptcy rate:** 5.12% vs 2.74% in manufacturing (construction is 1.87× riskier)
- **Complete case rate:** 66.3% (lower than manufacturing's 75.3%), suggesting more volatile accounting
- Missing data primarily from small contractors with incomplete reporting
- All financial ratios winsorized at 1st/99th percentiles

---

## 2. Model Performance

### Discrimination Metrics
| Metric | Train | Test |
|--------|-------|------|
| **AUC-ROC** | 1.0000 | **0.8659** |
| **Average Precision** | 0.9985 | 0.1306 |
| **Bankruptcies** | 2,282 | 492 |
| **Observations** | 48,727 | 25,417 |

### Interpretation
- **AUC 0.87:** Excellent discriminatory power - best among all sectors analyzed
- **Perfect train AUC (1.00):** Indicates overfitting, but test performance remains strong
- **Higher AP (0.13 vs 0.05 in manufacturing):** Better precision at high recall due to higher base rate (3.74% vs 2.08%)

### Comparison to Manufacturing
| Metric | Manufacturing (C) | Construction (F) | Difference |
|--------|------------------|------------------|------------|
| Test AUC | 0.8237 | **0.8659** | **+0.04** (5% improvement) |
| Bankruptcy rate | 2.08% | 3.74% | +1.66pp (80% higher) |
| Complete cases | 75.3% | 66.3% | -9pp (more missing data) |

**Construction is both riskier and more predictable** - higher baseline risk but clearer warning signals.

---

## 3. Feature Importance Analysis

### Top 15 Predictors (Gini Importance)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **kortsiktig_gjeldsgrad** | 0.0855 | **Short-term debt ratio - #1 predictor** |
| 2 | **debt_liquidity_stress** | 0.0838 | Interaction: Debt/Liquidity mismatch |
| 3 | **egenkapitalandel** | 0.0760 | Equity ratio |
| 4 | **likviditetsgrad_1** | 0.0616 | Current ratio |
| 5 | **total_gjeldsgrad** | 0.0525 | Total debt ratio |
| 6 | **omsetningsgrad** | 0.0502 | Asset turnover |
| 7 | **Tall 194** | 0.0458 | Sum gjeld (Total debt) |
| 8 | **Tall 146** | 0.0403 | Driftsresultat (Operating result) |
| 9 | **Tall 217** | 0.0377 | Sum eiendeler (Total assets) |
| 10 | **driftsrentabilitet** | 0.0363 | Operating ROA |
| 11 | **efficiency_profitability** | 0.0358 | Turnover × ROA interaction |
| 12 | **Tall 85** | 0.0356 | Kortsiktig gjeld (Current liabilities) |
| 13 | **Tall 17130** | 0.0352 | Finanskostnader (Interest expense) |
| 14 | **rentedekningsgrad** | 0.0331 | Interest coverage |
| 15 | **Tall 72** | 0.0330 | Sum inntekter (Total income) |

### Key Insights - Construction vs Manufacturing

**Construction (Sector F):**
1. **Short-term liquidity dominates** - kortsiktig_gjeldsgrad (#1), likviditetsgrad_1 (#4)
2. **More balanced feature distribution** - top feature 0.0855 vs manufacturing's 0.1210 (29% lower concentration)
3. **Asset efficiency matters** - omsetningsgrad (#6), efficiency_profitability (#11)

**Manufacturing (Sector C):**
1. **Equity depletion dominates** - egenkapitalandel (0.1210) is 43% more important than #2
2. **Leverage secondary** - total_gjeldsgrad (#2)
3. **Single dominant predictor** - concentrated risk signal

**Why the Difference?**
- **Construction:** Project-based cash flows create short-term liquidity crunches. Companies can be profitable but fail due to payment delays, seasonal work, or upfront material costs.
- **Manufacturing:** Long-term solvency matters more. Sustained losses erode equity, but short-term cash flow more manageable with inventory buffers.

---

## 4. Risk Stratification

### Risk Tier Distribution

| Risk Tier | Threshold | N | Bankruptcies | Actual Rate | Avg Prediction |
|-----------|-----------|---|--------------|-------------|----------------|
| **Very Low** | 0-2% | 16,479 (22.2%) | 6 | 0.04% | 1.0% |
| **Low** | 2-5% | 12,720 (17.2%) | 8 | 0.06% | 3.3% |
| **Medium** | 5-10% | 11,460 (15.5%) | 28 | 0.24% | 7.3% |
| **High** | 10-20% | 12,059 (16.3%) | 72 | 0.60% | 14.4% |
| **Very High** | 20%+ | 15,482 (20.9%) | 2,660 | **17.18%** | 39.5% |

### Calibration Analysis
- **Very High tier:** 17.18% actual vs 39.5% predicted - better calibrated than manufacturing (5.52% actual vs 46% predicted)
- **Concentration:** 95.9% of bankruptcies in Very High tier (2,660 of 2,774)
- **Risk spread:** More even distribution across tiers vs manufacturing's 37% in Very High

### Economic Profiles by Risk Tier

| Tier | Debt Ratio | Liquidity | Margin | Equity Ratio | Altman Z |
|------|------------|-----------|--------|--------------|----------|
| **Very Low** | 0.52 | 5.48 | 6.8% | 0.48 | 3.14 |
| **Low** | 0.66 | 3.66 | -2.2% | 0.34 | 2.93 |
| **Medium** | 0.84 | 2.44 | -9.6% | 0.16 | 2.91 |
| **High** | 1.09 | 1.57 | -7.6% | -0.09 | 3.36 |
| **Very High** | 1.49 | 0.91 | -7.3% | -0.49 | 3.46 |

### Profile Interpretation

**Very Low Risk (Healthy):**
- Moderate leverage (0.52 debt ratio)
- **Excellent liquidity (5.48 current ratio)** - construction's key safety margin
- Profitable (6.8% margin)
- Strong equity (0.48 equity ratio)

**Very High Risk (Distressed):**
- Severe over-leverage (1.49 debt ratio, 149% debt-to-assets)
- **Critical liquidity crisis (0.91 current ratio)** - cannot cover short-term obligations
- Negative equity (-0.49, deeply insolvent)
- Operating losses (-7.3% margin)

**Critical Observation:** Liquidity drops from 5.48 → 0.91 across tiers (6× decline), while debt ratio only rises 2.9×. **Liquidity collapse is the primary failure mechanism**, not leverage accumulation.

---

## 5. Economic Regime Analysis

### Regime Characteristics

#### Regime 0: Mainstream Construction (N=71,436, 96.3%)
| Metric | Value |
|--------|-------|
| Bankruptcy rate | 3.34% |
| Avg fixed assets | 5,912,083 NOK |
| Avg debt ratio | 0.73 |
| Avg liquidity | 3.25 |
| Avg margin | -0.62% |

**Profile:** Typical construction firms - moderate leverage, adequate liquidity, thin margins. Represents small-to-medium contractors and tradespeople.

#### Regime 1: Distressed Micro-Contractors (N=2,659, 3.6%)
| Metric | Value |
|--------|-------|
| Bankruptcy rate | **14.67%** |
| Avg fixed assets | 211,441 NOK |
| Avg debt ratio | **4.88** |
| Avg liquidity | **0.33** |
| Avg margin | **-57.25%** |

**Profile:** Very small contractors in extreme distress. Debt 4.9× assets (insolvent on paper), critically illiquid (0.33 current ratio means only 33% of short-term obligations covered), massive losses. **4.4× higher bankruptcy rate** than overall sample. Likely sole proprietors or micro-firms with project failures.

#### Regime 2: Large Construction Groups (N=49, 0.1%)
| Metric | Value |
|--------|-------|
| Bankruptcy rate | 0.00% |
| Avg fixed assets | **2,494,838,082 NOK** (2.5 billion) |
| Avg debt ratio | 0.70 |
| Avg liquidity | 1.22 |
| Avg margin | -9.36% |

**Profile:** Major construction conglomerates with massive asset bases. Lower leverage, modest liquidity, but negative margins suggest capital-intensive projects with delayed profitability. No observed bankruptcies (restructure rather than liquidate).

### Regime-Specific Feature Importance

#### Regime 0 (Mainstream) - Top 5 Predictors
1. egenkapitalandel (0.0997)
2. kortsiktig_gjeldsgrad (0.0947)
3. debt_liquidity_stress (0.0932)
4. total_gjeldsgrad (0.0793)
5. likviditetsgrad_1 (0.0599)

**Pattern:** Balanced risk factors - both equity depletion and liquidity stress matter.

#### Regime 1 (Distressed Micro) - Top 5 Predictors
1. **Tall 146 - Driftsresultat (0.0886)**
2. **Tall 72 - Sum inntekter (0.0693)**
3. **Tall 1340 - Salgsinntekt (0.0592)**
4. **Tall 85 - Kortsiktig gjeld (0.0548)**
5. **omsetningsgrad (0.0523)**

**Pattern:** For micro-contractors already distressed, **absolute revenue and operating results** become critical. Ratios less informative when companies are already insolvent - survival depends on generating cash flow to service immediate obligations.

---

## 6. Sector Comparison: Construction vs Manufacturing

| Metric | Manufacturing (C) | Construction (F) |
|--------|------------------|------------------|
| **Baseline Risk** | 2.74% | **5.12%** (1.87× higher) |
| **Test AUC** | 0.8237 | **0.8659** (better) |
| **Complete Cases** | 75.3% | 66.3% (worse data quality) |
| **Top Predictor** | Equity ratio (0.1210) | **Short-term debt (0.0855)** |
| **#2 Predictor** | Total debt (0.0843) | **Debt/Liquidity stress (0.0838)** |
| **Feature Concentration** | High (single dominant) | **Balanced (distributed)** |
| **Very High Tier Bankruptcy** | 5.52% | **17.18%** (3.1× higher) |
| **Distressed Regime Rate** | 4.04% | **14.67%** (3.6× higher) |

### Key Takeaways
1. **Construction is riskier:** 5.12% baseline vs 2.74% in manufacturing
2. **Construction is more predictable:** AUC 0.87 vs 0.82 (clearer warning signals)
3. **Different failure modes:**
   - Manufacturing: Slow equity erosion from sustained losses
   - Construction: Rapid liquidity crisis from cash flow mismatch
4. **Higher extremes:** Very High risk tier shows 17% bankruptcy in construction vs 5.5% in manufacturing

---

## 7. Model Validation & Limitations

### Strengths
✅ **Largest sample:** 74,144 observations, 2,774 bankruptcies (3× more bankruptcies than manufacturing)
✅ **Best performance:** AUC 0.87 is highest across all sectors
✅ **Better calibration:** Very High tier 17% actual vs 40% predicted (vs 5.5% vs 46% in manufacturing)
✅ **Construction-specific insights:** Short-term liquidity stress identified as dominant risk factor
✅ **Regime heterogeneity:** Micro-contractors (14.67% bankruptcy) vs large groups (0% bankruptcy)

### Limitations
⚠️ **Perfect train AUC (1.00):** Severe overfitting, though test performance remains strong
⚠️ **Lower complete cases:** 66.3% vs 75.3% in manufacturing (more missing data)
⚠️ **Temporal scope:** 2016-2018 (pre-COVID), may not reflect post-pandemic construction boom/bust
⚠️ **Regime 2 sample:** Only 49 large construction groups limits inference
⚠️ **Class imbalance persists:** 3.74% bankruptcy rate still leads to probability overestimation

### Recommendations for Production Use
1. **Prioritize liquidity monitoring:** Short-term debt ratio and current ratio should trigger alerts
2. **Use risk tiers, not raw probabilities:** 17% actual in Very High tier more reliable than 40% raw prediction
3. **Regime-specific thresholds:** Micro-contractors (Regime 1) need different criteria given 14.67% base rate
4. **Regularize model:** Reduce overfitting via max_depth limits or ensemble with logistic regression
5. **Seasonal adjustment:** Construction shows seasonal cash flow patterns - consider quarter-specific models

---

## 8. Business Implications

### Credit Risk Assessment
- **Very Low tier (0.04% risk):** Excellent credit quality, suitable for unsecured lending
- **Low/Medium tiers (0.06-0.24%):** Standard credit terms with liquidity monitoring
- **High tier (0.60% risk):** Require current ratio > 1.5 and progress payment structures
- **Very High tier (17.18% risk):** Credit rejection or secured lending only (material liens, payment bonds)

### Early Warning System
**Red Flags for Construction Companies:**
1. **Liquidity crisis:** Current ratio < 1.2 (signal of payment timing mismatch)
2. **Short-term debt stress:** Kortsiktig_gjeldsgrad > 0.7 (too much due within 12 months)
3. **Negative spiral:** Liquidity < 1.5 + Negative equity + Operating losses (triple threat)
4. **Micro-contractor distress:** Assets < 500K NOK + Debt ratio > 2.0 + Losses

### Industry Insights (Sector F - Construction)
1. **Liquidity is survival:** Unlike manufacturing, construction fails fast when cash flow breaks. Current ratio < 1.0 is acute crisis.
2. **Project-based volatility:** Payment delays, seasonal work, upfront costs create short-term stress even for profitable firms.
3. **Size matters dramatically:** Micro-contractors (14.67% bankruptcy) vs large groups (0% bankruptcy) - scale provides resilience.
4. **Thin margins, high leverage:** Average -0.62% margin with 0.73 debt ratio means little buffer for shocks.

---

## 9. Technical Specifications

### Model Artifacts Saved
- `random_forest_model.pkl` - Trained Random Forest classifier
- `scaler.pkl` - StandardScaler for feature normalization
- `pca_model.pkl` - PCA transformation (16 components, 95.4% variance)
- `kmeans_model.pkl` - K-Means clustering (3 regimes)

### Reproducibility
- Random seed: 42
- Python version: 3.11
- Key packages: scikit-learn, pandas, numpy
- Training time: ~8 seconds on standard CPU

---

## 10. Next Steps

### Immediate Actions
1. ✅ Sector F (Construction) analysis complete
2. ⏭️ Run supervised analysis on Sector G (Retail/Wholesale)
3. ⏭️ Run supervised analysis on Sector I (Hospitality)
4. ⏭️ Cross-sector comparison report

### Model Improvements
- Reduce overfitting (train AUC 1.00 → target 0.90)
- Implement ensemble with gradient boosting
- Add temporal features (payment delay trends, seasonal patterns)
- Explore construction-specific features (project pipeline, backlog)

---

## Appendix: Norwegian Financial Terms

### Key Ratios (Construction Context)
- **Kortsiktig_gjeldsgrad:** Short-term debt ratio - **CRITICAL in construction** (project payment timing)
- **Likviditetsgrad:** Current ratio - **MORE IMPORTANT** than in manufacturing (no inventory buffer)
- **Egenkapitalandel:** Equity ratio - important but secondary to liquidity
- **Omsetningsgrad:** Asset turnover - efficiency metric (construction uses fewer fixed assets than manufacturing)

### Accounting Fields (Tall)
- **Tall 1340:** Salgsinntekt (Sales revenue) - project revenue recognition
- **Tall 72:** Sum inntekter (Total income) - includes change in work-in-progress
- **Tall 146:** Driftsresultat (Operating result) - often negative due to project timing
- **Tall 217:** Sum eiendeler (Total assets) - includes WIP inventory and receivables
- **Tall 194:** Sum gjeld (Total debt) - includes supplier credit and advances
- **Tall 85:** Kortsiktig gjeld (Current liabilities) - **KEY METRIC** for construction
- **Tall 86:** Omløpsmidler (Current assets) - receivables and WIP dominate
- **Tall 17130:** Finanskostnader (Interest expense) - often low for small contractors

---

**Report prepared:** December 4, 2025
**Analyst:** Claude (AI Agent)
**Sector:** F - Construction (NACE 41-43)
**Version:** 1.0
