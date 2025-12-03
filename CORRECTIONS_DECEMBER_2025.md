# Financial Ratios Corrections - December 3, 2025

## Summary

All financial ratios have been corrected to comply with Norwegian accounting standards. The feature dataset has been rebuilt, and all old predictions have been archived.

---

## Corrections Made

### 1. driftsmargin - CORRECTED FORMULA ✅

**Issue:** Used total income instead of operating revenue

**Old Formula (Incorrect):**
```python
driftsmargin = Tall 146 / Tall 72
             = Driftsresultat / Sum inntekter
```

**New Formula (Correct):**
```python
driftsmargin = Tall 146 / Tall 1340
             = Driftsresultat / Salgsinntekt
```

**Reason:** Operating margin should use operating revenue (Salgsinntekt), not total income which includes financial income. This aligns with Norwegian accounting standards.

---

### 2. totalkapitalrentabilitet → driftsrentabilitet - RENAMED ✅

**Issue:** Metric was misnamed given data limitations

**Old Name:** `totalkapitalrentabilitet`
**New Name:** `driftsrentabilitet`

**Formula (unchanged):**
```python
driftsrentabilitet = Tall 146 / (Tall 217 + Tall 194)
                   = Driftsresultat / Totalkapital
```

**Reason for Rename:**
- Standard Norwegian totalkapitalrentabilitet = (Årsresultat + Finanskostnader) / Totalkapital
- We use: Driftsresultat / Totalkapital (Operating ROA)
- Data limitation: Årsresultat (net income) not available in dataset
- New name "driftsrentabilitet" accurately describes what we calculate

**Academic Justification:**
- Operating-based profitability metrics are common in bankruptcy research (Beaver 1966, Altman 1968)
- More comparable across companies (excludes tax rate variations)
- Focuses on core business operating performance

---

### 3. likviditetsgrad_2 - REMOVED ✅

**Issue:** Redundant feature due to missing inventory data

**Standard Formula:**
```python
likviditetsgrad_2 = (Omløpsmidler - Varelager) / Kortsiktig gjeld
```

**What we had:**
```python
likviditetsgrad_2 = likviditetsgrad_1  # Exact duplicate
```

**Action:** Removed entirely to avoid redundancy and multicollinearity

**Reason:** Dataset does not contain inventory (varelager) data, making true quick ratio calculation impossible. Feature was identical to likviditetsgrad_1.

---

## Impact Summary

| Change | Impact | Priority |
|--------|--------|----------|
| driftsmargin correction | Moderate - May affect profitability analysis | MODERATE |
| Rename to driftsrentabilitet | High - Clarifies what metric represents | CRITICAL |
| Remove likviditetsgrad_2 | Low - Was redundant | LOW |

**Feature Count:** Reduced from 40 to 38 features

---

## Files Updated

### 1. Code Changes

**File:** `INF4090/data/features/build_features.py`

**Lines 148-157:** Fixed driftsmargin
```python
# OLD:
df_features['driftsmargin'] = safe_divide(df['Tall 146'], df['Tall 72'])

# NEW:
df_features['driftsmargin'] = safe_divide(df['Tall 146'], df['Tall 1340'])
```

**Lines 159-170:** Renamed totalkapitalrentabilitet
```python
# OLD:
df_features['totalkapitalrentabilitet'] = safe_divide(df['Tall 146'], df_features['_total_assets'])

# NEW:
df_features['driftsrentabilitet'] = safe_divide(df['Tall 146'], df_features['_total_assets'])
```

**Lines 96-100:** Removed likviditetsgrad_2
```python
# DELETED:
# df_features['likviditetsgrad_2'] = df_features['likviditetsgrad_1'].copy()
```

---

### 2. Data Regenerated

**File:** `INF4090/data/features/feature_dataset_v1.parquet` (101.68 MB)
**File:** `INF4090/data/features/feature_dataset_v1.csv` (1,054.18 MB)

**Rebuilt:** December 3, 2025 at 12:15

**Verification:**
- ✅ `driftsrentabilitet` exists (renamed from totalkapitalrentabilitet)
- ✅ `totalkapitalrentabilitet` removed
- ✅ `likviditetsgrad_2` removed
- ✅ Dataset shape: (280,840 rows, 117 columns)
- ✅ 38 engineered features + base columns

---

### 3. Documentation Updated

**Created:** `INF4090/data/features/feature_specification.md`
- Complete specification of all 38 features
- Norwegian accounting standards compliance
- Data limitations documented
- Changelog with corrections

**Auto-generated:**
- `feature_calculation_log.md` - Updated with new formulas
- `feature_statistics.txt` - Statistics for corrected features

---

### 4. Predictions Archived

**Old predictions moved to:** `Legacy/predictions_old_ratios/`

**Contents archived:**
- `2018_only/` - Baseline model using only 2018 data
- `all_years/` - Model using all three years
- `unsupervised_all_features/` - Comprehensive unsupervised model
- `sector_specific/` - Sector-specific bankruptcy models (C, F, G, I)
- `statistics/` - Filing behavior and size analysis

**New predictions folder:** `INF4090/predictions/` (empty, ready for fresh analysis)

---

## Data Limitations Documented

### 1. Årsresultat (Net Income) NOT AVAILABLE
- Affects: ROA calculation
- Workaround: Use driftsrentabilitet (Operating ROA)
- Justification: Common in bankruptcy research, more comparable

### 2. Varelager (Inventory) NOT AVAILABLE
- Affects: Quick ratio calculation
- Workaround: Removed likviditetsgrad_2 entirely
- Impact: Minimal (was redundant)

### 3. Potential Data Leakage in antall_år_levert
- Issue: Values 4, 5, 6 (should be max 3) have 100% bankruptcy rate
- Discovered: During sector-specific analysis
- Recommendation: Filter to values 0-3 before modeling

---

## Norwegian Accounting Standards Compliance

All ratios now comply with:
- **Regnskapsloven** (Norwegian Accounting Act)
- **Norsk RegnskapsStiftelse (NRS)** standards
- **Kinserdal (2020)** "Analyse av årsregnskap"

### Verified Correct (11 ratios):

1. ✅ likviditetsgrad_1 - Omløpsmidler / Kortsiktig gjeld
2. ✅ total_gjeldsgrad - Total gjeld / Totalkapital
3. ✅ langsiktig_gjeldsgrad - Langsiktig gjeld / Totalkapital
4. ✅ kortsiktig_gjeldsgrad - Kortsiktig gjeld / Totalkapital
5. ✅ egenkapitalandel - Egenkapital / Totalkapital
6. ✅ **driftsmargin** - Driftsresultat / Salgsinntekt (CORRECTED)
7. ✅ **driftsrentabilitet** - Driftsresultat / Totalkapital (RENAMED)
8. ✅ omsetningsgrad - Salgsinntekt / Totalkapital (informal name for "totalkapitalens omløpshastighet")
9. ✅ rentedekningsgrad - Driftsresultat / Finanskostnader
10. ✅ altman_z_score - Standard Altman formula
11. ❌ ~~likviditetsgrad_2~~ - REMOVED (was incorrect)

---

## Next Steps

### For Modeling:
1. Use new corrected dataset: `feature_dataset_v1.parquet`
2. Be aware: `driftsrentabilitet` replaces `totalkapitalrentabilitet`
3. Note: Only 38 features now (was 40)
4. Filter `antall_år_levert` to 0-3 only (avoid data leakage)

### For Thesis:
1. Include data limitations section explaining:
   - Why using operating-based ROA (årsresultat unavailable)
   - Why likviditetsgrad_2 was removed (inventory data unavailable)
2. Add terminology table mapping informal → formal Norwegian terms
3. Reference corrections made (show academic rigor)

---

## References Used

### Norwegian Accounting Standards:
1. Regnskapsloven - Norwegian Accounting Act
2. Norsk RegnskapsStiftelse (NRS)
3. Kinserdal, F. (2020). *Analyse av årsregnskap*. Cappelen Damm Akademisk.

### Bankruptcy Prediction Literature:
1. Beaver, W.H. (1966). Financial Ratios as Predictors of Failure. *Journal of Accounting Research*, 4, 71-111.
2. Altman, E.I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589-609.

---

## Verification Checklist

- [x] build_features.py updated with corrections
- [x] Feature dataset rebuilt (feature_dataset_v1.parquet)
- [x] feature_specification.md created
- [x] feature_calculation_log.md regenerated
- [x] Old predictions moved to Legacy folder
- [x] New empty predictions folder created
- [x] Verified driftsrentabilitet exists in dataset
- [x] Verified totalkapitalrentabilitet removed from dataset
- [x] Verified likviditetsgrad_2 removed from dataset
- [x] Dataset shape correct: 280,840 rows, 117 columns

---

**Date:** December 3, 2025
**Status:** COMPLETE
**Impact:** All financial ratios now comply with Norwegian accounting standards
**Next:** Ready for fresh modeling with corrected features
