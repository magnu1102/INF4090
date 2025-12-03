# Feature Specification - Norwegian Bankruptcy Prediction
## Compliance with Norwegian Accounting Standards

**Last Updated:** December 3, 2025
**Total Features:** 38 (corrected from 40)
**Dataset:** feature_dataset_v1.parquet

---

## Important Notes on Corrections

**Three critical corrections made to align with Norwegian accounting standards:**

1. **driftsmargin** - Fixed denominator from Tall 72 (Sum inntekter) to Tall 1340 (Salgsinntekt)
2. **totalkapitalrentabilitet** → **driftsrentabilitet** - Renamed for accuracy (uses operating income, not net income)
3. **likviditetsgrad_2** - Removed (was redundant duplicate due to missing inventory data)

**Data Limitations:**
- Årsresultat (net income) NOT AVAILABLE in dataset - affects ROA calculation
- Varelager (inventory) NOT AVAILABLE - affects quick ratio calculation

---

## CATEGORY 1: FINANCIAL RATIOS (11 features)

### 1.1 Liquidity Ratios (1 feature)

#### likviditetsgrad_1
- **Norwegian Term:** Likviditetsgrad 1 (Current Ratio)
- **Formula:** `Tall 194 / Tall 85` = Omløpsmidler / Kortsiktig gjeld
- **Standard:** NRS, Regnskapsloven
- **Interpretation:** >1 indicates ability to pay short-term obligations
- **References:** Beaver (1966), Altman (1968)
- **Status:** ✅ Correct per Norwegian standards

**REMOVED: likviditetsgrad_2**
- Previously identical to likviditetsgrad_1
- Standard formula requires inventory data: (Omløpsmidler - Varelager) / Kortsiktig gjeld
- Removed to avoid redundancy and multicollinearity

---

### 1.2 Leverage/Solvency Ratios (4 features)

#### total_gjeldsgrad
- **Norwegian Term:** Gjeldsgrad (Debt Ratio)
- **Formula:** `(Tall 86 + Tall 85) / (Tall 217 + Tall 194)` = Total gjeld / Totalkapital
- **Standard:** Uses debt-to-assets definition (0-1 scale), not debt-to-equity
- **Interpretation:** Higher values = more leveraged, lower solvency
- **References:** Altman (1968), Zmijewski (1984)
- **Status:** ✅ Correct

#### langsiktig_gjeldsgrad
- **Norwegian Term:** Langsiktig gjeldsgrad (Long-term Debt Ratio)
- **Formula:** `Tall 86 / (Tall 217 + Tall 194)` = Langsiktig gjeld / Totalkapital
- **Interpretation:** Proportion of assets financed by long-term debt
- **Status:** ✅ Correct

#### kortsiktig_gjeldsgrad
- **Norwegian Term:** Kortsiktig gjeldsgrad (Short-term Debt Ratio)
- **Formula:** `Tall 85 / (Tall 217 + Tall 194)` = Kortsiktig gjeld / Totalkapital
- **Interpretation:** Proportion of assets financed by short-term debt
- **Status:** ✅ Correct

#### egenkapitalandel
- **Norwegian Term:** Egenkapitalandel (Equity Ratio)
- **Formula:** `1 - total_gjeldsgrad` = Egenkapital / Totalkapital
- **Standard:** Regnskapsloven §3-2
- **Interpretation:** Higher values = better solvency. Should equal 1 - gjeldsgrad
- **Status:** ✅ Correct

---

### 1.3 Profitability Ratios (2 features)

#### driftsmargin
- **Norwegian Term:** Driftsmargin (Operating Margin)
- **Formula:** `Tall 146 / Tall 1340` = Driftsresultat / Salgsinntekt
- **Standard:** Norwegian accounting standards
- **Interpretation:** Operating profit per unit of sales revenue
- **Status:** ✅ **CORRECTED** - Changed from Tall 72 to Tall 1340

**Correction Details:**
- **Old (incorrect):** Tall 146 / Tall 72 (Driftsresultat / Sum inntekter)
- **New (correct):** Tall 146 / Tall 1340 (Driftsresultat / Salgsinntekt)
- **Reason:** Sum inntekter includes financial income; operating margin should use operating revenue only

#### driftsrentabilitet
- **Norwegian Term:** Driftsrentabilitet (Operating Return on Assets)
- **Formula:** `Tall 146 / (Tall 217 + Tall 194)` = Driftsresultat / Totalkapital
- **Interpretation:** Operating profitability relative to total assets
- **References:** Altman (1968), Beaver (1966)
- **Status:** ✅ **RENAMED** for accuracy

**Renaming Details:**
- **Old name:** totalkapitalrentabilitet
- **New name:** driftsrentabilitet
- **Reason:** Standard totalkapitalrentabilitet = (Årsresultat + Finanskostnader) / Totalkapital
- **Data Limitation:** Årsresultat (net income) not available in dataset
- **What we calculate:** Driftsresultat / Totalkapital = Operating ROA (EBIT / Assets)
- **Justification:** Operating-based profitability measures are common in bankruptcy prediction literature and more comparable across companies (no tax rate differences)

**Norwegian Standard Reference:**
According to Kinserdal (2020) "Analyse av årsregnskap":
```
Totalkapitalrentabilitet = (Årsresultat + Finanskostnader) / Totalkapital
```
Finanskostnader added back because totalkapital includes both equity and debt; return should include returns to both equity holders (årsresultat) and debt holders (finanskostnader).

---

### 1.4 Efficiency Ratios (1 feature)

#### omsetningsgrad
- **Norwegian Term (Standard):** Totalkapitalens omløpshastighet (Asset Turnover)
- **Informal Name:** omsetningsgrad (used in code for brevity)
- **Formula:** `Tall 1340 / (Tall 217 + Tall 194)` = Salgsinntekt / Totalkapital
- **Standard:** Kinserdal (2020), standard financial analysis
- **Interpretation:** How efficiently assets generate sales revenue
- **Status:** ✅ Formula correct; name is informal shorthand

**Terminology Note:**
- Code uses "omsetningsgrad" (informal)
- Norwegian standard: "totalkapitalens omløpshastighet" or "kapitalomsetning"
- Issue: "-grad" (degree) is incorrect suffix; should be "-hastighet" (velocity) or "-omsetning" (turnover)
- Formula is correct per Norwegian standards

---

### 1.5 Coverage Ratios (1 feature)

#### rentedekningsgrad
- **Norwegian Term:** Rentedekningsgrad (Interest Coverage Ratio)
- **Formula:** `Tall 146 / Tall 17130` = Driftsresultat / Sum finanskostnader
- **Standard:** Times Interest Earned ratio
- **Interpretation:** How many times can company cover interest payments from operating income
- **Status:** ✅ Correct

---

### 1.6 Composite Scores (1 feature)

#### altman_z_score
- **Model:** Altman Z-Score (Simplified for private companies)
- **Formula:** `0.717*X1 + 3.107*X3 + 0.420*X4 + 0.998*X5`
- **Components:**
  - X1 = Working Capital / Total Assets
  - X3 = EBIT / Total Assets
  - X4 = Equity / Total Debt
  - X5 = Sales / Total Assets
- **Note:** X2 (Retained Earnings / Total Assets) omitted - data not available
- **References:** Altman (1968), revised for private companies (1983)
- **Interpretation:**
  - Z > 2.9: Safe zone
  - 1.23 < Z < 2.9: Grey zone
  - Z < 1.23: Distress zone
- **Status:** ✅ Uses correct Altman formula

---

## CATEGORY 2: TEMPORAL FEATURES (10 features)

### 2.1 Growth Rates (6 features)

#### omsetningsvekst_1617 / omsetningsvekst_1718
- **Formula:** `(Salgsinntekt_year2 - Salgsinntekt_year1) / Salgsinntekt_year1`
- **Interpretation:** Revenue growth rate (negative = decline)

#### aktivavekst_1617 / aktivavekst_1718
- **Formula:** `(Totalkapital_year2 - Totalkapital_year1) / Totalkapital_year1`
- **Interpretation:** Asset growth rate

#### gjeldsvekst_1617 / gjeldsvekst_1718
- **Formula:** `(Total_gjeld_year2 - Total_gjeld_year1) / Total_gjeld_year1`
- **Interpretation:** Debt growth rate (rapid growth can signal distress)

---

### 2.2 Trend Indicators (3 features - Binary)

#### fallende_likviditet
- **Definition:** Likviditetsgrad decreased in both 2016→2017 and 2017→2018
- **Interpretation:** 1 = deteriorating liquidity trend

#### konsistent_underskudd
- **Definition:** Negative driftsresultat in all available years
- **Interpretation:** 1 = persistent operating losses

#### økende_gjeldsgrad
- **Definition:** Gjeldsgrad increased in both consecutive periods
- **Interpretation:** 1 = rising leverage trend

---

### 2.3 Volatility Measures (1 feature)

#### omsetningsvolatilitet
- **Formula:** Standard deviation of revenue across available years
- **Interpretation:** Higher values = more volatile revenue (potential instability)

---

## CATEGORY 3: MISSINGNESS FEATURES (7 features)

**Rationale:** In bankruptcy prediction, missing data itself is predictive (76.5% of bankrupt companies didn't file 2018 data)

#### levert_alle_år
- **Definition:** 1 if company filed financial statements in all three years (2016, 2017, 2018)
- **Status:** Binary (0/1)

#### levert_2018
- **Definition:** 1 if company filed financial statements in 2018
- **Status:** Binary (0/1)

#### antall_år_levert
- **Definition:** Count of years company filed financial statements (0-3)
- **Status:** Ordinal (0, 1, 2, 3)

**WARNING:** antall_år_levert may contain values >3 that encode bankruptcy status (data leakage)
- This issue was discovered in sector-specific analysis
- Values 4, 5, 6 have 100% bankruptcy rate
- **Recommendation:** Filter to only values 0-3 before modeling

#### regnskapskomplett
- **Definition:** 1 if all 9 critical accounting fields (Tall fields) are present
- **Fields:** Tall 1340, 7709, 72, 217, 194, 86, 85, 146, 17130
- **Status:** Binary (0/1)

#### kan_ikke_beregne_likviditet
- **Definition:** 1 if missing data prevents calculating likviditetsgrad
- **Status:** Binary (0/1)

#### kan_ikke_beregne_gjeldsgrad
- **Definition:** 1 if missing data prevents calculating gjeldsgrad
- **Status:** Binary (0/1)

---

## CATEGORY 4: COMPANY CHARACTERISTICS (7 features)

### Basic Characteristics (4 features)

#### selskapsalder
- **Formula:** `(2018 - Stiftelsesdato year)`
- **Interpretation:** Company age in years. Younger companies have higher bankruptcy risk (Ohlson 1980)

#### nytt_selskap
- **Definition:** 1 if company founded within 3 years of observation year
- **Status:** Binary (0/1)

#### log_totalkapital
- **Formula:** `log(Totalkapital)`
- **Interpretation:** Log-transformed total assets. Larger companies generally lower bankruptcy risk

#### log_omsetning
- **Formula:** `log(Salgsinntekt)`
- **Interpretation:** Log-transformed revenue

---

### Auditor Change (3 features - Binary)

#### byttet_revisor_1617 / byttet_revisor_1718
- **Definition:** 1 if company changed auditor between consecutive years
- **Interpretation:** Auditor changes can signal problems

#### byttet_revisor_noensinne
- **Definition:** 1 if company changed auditor at any point in observation period
- **Status:** Binary (0/1)

---

## CATEGORY 5: WARNING SIGNALS (5 features - Binary)

**All binary flags (0/1) indicating financial distress**

#### negativ_egenkapital
- **Definition:** Egenkapitalandel < 0 (negative equity)
- **Interpretation:** Liabilities exceed assets - technically insolvent

#### sterkt_overbelånt
- **Definition:** Total_gjeldsgrad > 0.80 (80%+ debt)
- **Interpretation:** Highly leveraged, low solvency buffer

#### kan_ikke_dekke_renter
- **Definition:** Rentedekningsgrad < 1 (operating income < interest expense)
- **Interpretation:** Cannot cover interest payments from operations

#### lav_likviditet
- **Definition:** Likviditetsgrad_1 < 1 (current assets < current liabilities)
- **Interpretation:** Cannot cover short-term obligations

#### driftsunderskudd
- **Definition:** Driftsmargin < 0 (negative operating margin)
- **Interpretation:** Operating losses

---

## DATA QUALITY & LIMITATIONS

### Missing Data
- **Feature completeness varies:** 77-96% of observations have each feature
- **Worst coverage:**
  - driftsmargin: 77.5% (63,273 missing)
  - omsetningsgrad: 78.1% (61,418 missing)
  - Growth features: 65-80% (require consecutive years)

### Key Data Limitations

#### 1. No Net Income (Årsresultat)
**Impact:** Cannot calculate true totalkapitalrentabilitet per Norwegian standards

**Standard Formula:**
```
Totalkapitalrentabilitet = (Årsresultat + Finanskostnader) / Totalkapital
```

**What We Calculate:**
```
Driftsrentabilitet = Driftsresultat / Totalkapital
```

**Justification:**
- Operating-based metrics common in bankruptcy literature (Beaver 1966, Altman 1968)
- More comparable across companies (excludes tax rate variations)
- Focuses on core business performance

#### 2. No Inventory (Varelager)
**Impact:** Cannot calculate true quick ratio (likviditetsgrad_2)

**Standard Formula:**
```
Likviditetsgrad 2 = (Omløpsmidler - Varelager) / Kortsiktig gjeld
```

**Resolution:** Feature removed entirely (was redundant duplicate)

#### 3. Potential Data Leakage in antall_år_levert
**Issue:** Values 4, 5, 6 (should be max 3) have 100% bankruptcy rate
**Impact:** May artificially inflate model performance
**Recommendation:** Filter to values 0-3 only

---

## REFERENCES

### Norwegian Accounting Standards
1. **Regnskapsloven** - Norwegian Accounting Act
2. **Norsk RegnskapsStiftelse (NRS)** - Norwegian Accounting Standards Board
3. **Kinserdal, F.** (2020). *Analyse av årsregnskap*. Cappelen Damm Akademisk.

### Bankruptcy Prediction Literature
1. **Beaver, W.H.** (1966). Financial Ratios as Predictors of Failure. *Journal of Accounting Research*, 4, 71-111.
2. **Altman, E.I.** (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589-609.
3. **Altman, E.I.** (1983). Corporate Financial Distress. Wiley.
4. **Ohlson, J.A.** (1980). Financial Ratios and the Probabilistic Prediction of Bankruptcy. *Journal of Accounting Research*, 18(1), 109-131.
5. **Zmijewski, M.E.** (1984). Methodological Issues Related to the Estimation of Financial Distress Prediction Models. *Journal of Accounting Research*, 22, 59-82.

---

## CHANGELOG

### December 3, 2025 - Major Corrections
**Changes made to align with Norwegian accounting standards:**

1. **driftsmargin** - CORRECTED
   - Old: `Tall 146 / Tall 72` (Driftsresultat / Sum inntekter)
   - New: `Tall 146 / Tall 1340` (Driftsresultat / Salgsinntekt)
   - Reason: Operating margin should use operating revenue, not total income

2. **totalkapitalrentabilitet → driftsrentabilitet** - RENAMED
   - More accurate name given data limitations
   - Clearly indicates it's operating-based ROA
   - Documented limitation: årsresultat not available

3. **likviditetsgrad_2** - REMOVED
   - Was identical to likviditetsgrad_1
   - Required inventory data not available
   - Removed to avoid redundancy

**Impact:** Feature count reduced from 40 to 38

---

## SUMMARY TABLE - ALL 38 FEATURES

| Category | Feature | Type | Norwegian Standard | Status |
|----------|---------|------|-------------------|--------|
| **Financial Ratios** | | | | |
| Liquidity | likviditetsgrad_1 | Continuous | Likviditetsgrad 1 | ✅ |
| Leverage | total_gjeldsgrad | Continuous | Gjeldsgrad | ✅ |
| Leverage | langsiktig_gjeldsgrad | Continuous | Langsiktig gjeldsgrad | ✅ |
| Leverage | kortsiktig_gjeldsgrad | Continuous | Kortsiktig gjeldsgrad | ✅ |
| Leverage | egenkapitalandel | Continuous | Egenkapitalandel | ✅ |
| Profitability | driftsmargin | Continuous | Driftsmargin | ✅ CORRECTED |
| Profitability | driftsrentabilitet | Continuous | Driftsrentabilitet (Operating ROA) | ✅ RENAMED |
| Efficiency | omsetningsgrad | Continuous | Totalkapitalens omløpshastighet | ✅ (informal name) |
| Coverage | rentedekningsgrad | Continuous | Rentedekningsgrad | ✅ |
| Composite | altman_z_score | Continuous | Altman Z-Score | ✅ |
| **Temporal** | | | | |
| Growth | omsetningsvekst_1617 | Continuous | Revenue growth 2016→2017 | ✅ |
| Growth | omsetningsvekst_1718 | Continuous | Revenue growth 2017→2018 | ✅ |
| Growth | aktivavekst_1617 | Continuous | Asset growth 2016→2017 | ✅ |
| Growth | aktivavekst_1718 | Continuous | Asset growth 2017→2018 | ✅ |
| Growth | gjeldsvekst_1617 | Continuous | Debt growth 2016→2017 | ✅ |
| Growth | gjeldsvekst_1718 | Continuous | Debt growth 2017→2018 | ✅ |
| Trend | fallende_likviditet | Binary | Declining liquidity | ✅ |
| Trend | konsistent_underskudd | Binary | Persistent losses | ✅ |
| Trend | økende_gjeldsgrad | Binary | Increasing leverage | ✅ |
| Volatility | omsetningsvolatilitet | Continuous | Revenue volatility | ✅ |
| **Missingness** | | | | |
| Filing | levert_alle_år | Binary | Filed all years | ✅ ⚠️ Leakage risk |
| Filing | levert_2018 | Binary | Filed in 2018 | ✅ |
| Filing | antall_år_levert | Ordinal (0-3) | Years filed | ⚠️ Data leakage (values >3) |
| Completeness | regnskapskomplett | Binary | All fields present | ✅ |
| Missingness | kan_ikke_beregne_likviditet | Binary | Missing liquidity data | ✅ |
| Missingness | kan_ikke_beregne_gjeldsgrad | Binary | Missing leverage data | ✅ |
| **Company Characteristics** | | | | |
| Age | selskapsalder | Continuous | Company age (years) | ✅ |
| Age | nytt_selskap | Binary | Young company (<3 years) | ✅ |
| Size | log_totalkapital | Continuous | Log total assets | ✅ |
| Size | log_omsetning | Continuous | Log revenue | ✅ |
| Auditor | byttet_revisor_1617 | Binary | Auditor change 2016→2017 | ✅ |
| Auditor | byttet_revisor_1718 | Binary | Auditor change 2017→2018 | ✅ |
| Auditor | byttet_revisor_noensinne | Binary | Ever changed auditor | ✅ |
| **Warning Signals** | | | | |
| Distress | negativ_egenkapital | Binary | Negative equity | ✅ |
| Distress | sterkt_overbelånt | Binary | High leverage (>80%) | ✅ |
| Distress | kan_ikke_dekke_renter | Binary | Interest coverage <1 | ✅ |
| Distress | lav_likviditet | Binary | Current ratio <1 | ✅ |
| Distress | driftsunderskudd | Binary | Operating losses | ✅ |

**Total: 38 features** (11 ratios + 10 temporal + 6 missingness + 7 characteristics + 5 warnings - 1 removed)

---

**Document Status:** Updated and verified against Norwegian accounting standards
**Last Audit:** December 3, 2025
**Next Review:** Before thesis submission
