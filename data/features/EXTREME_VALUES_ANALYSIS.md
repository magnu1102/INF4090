# Extreme Values Analysis - Financial Ratios

**Date:** December 3, 2025
**Scope:** Sector C (Manufacturing), 34,223 observations
**Issue:** Extreme outlier values in calculated financial ratios

---

## Executive Summary

Financial ratios computed in `build_features.py` contain extreme outlier values (e.g., likviditetsgrad = 6,967,346, gjeldsgrad = 12,702,030) that corrupt analysis. Root cause identified: **near-zero and negative denominator values** in accounting data create mathematically valid but economically nonsensical ratios.

**Key findings:**
- 0.05-0.48% of observations have near-zero denominators (< 1,000 NOK)
- 0.14-0.51% have negative denominators (debt, assets, or revenue < 0)
- These edge cases produce extreme ratios that distort model training and risk profiles
- Most extreme cases occur in companies with minimal economic activity (1 NOK assets, 7 NOK debt)

---

## Root Cause Analysis

### Current Implementation

The `safe_divide()` function in `build_features.py` (lines 78-82):

```python
def safe_divide(numerator, denominator):
    """Divide but return NaN if denominator is 0 or NaN"""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
```

**What it does:**
- Handles exact zero denominators by converting inf → NaN
- Does NOT handle near-zero denominators
- Does NOT handle negative denominators

**Problem:** When denominator is 1 NOK and numerator is 6,967,346 NOK, the ratio is 6,967,346 (mathematically correct but economically nonsensical).

---

## Quantitative Evidence

### Denominator Analysis (Sector C)

#### Tall 85 (Kortsiktig gjeld) - Used for likviditetsgrad
- **Zero values:** 158 (0.48%)
- **Negative values:** 168 (0.51%)
- **Near-zero (< 1,000 NOK):** 158 (0.48%)
- **Distribution:**
  - Min: -39,240,955 NOK
  - 1st percentile: 2 NOK
  - Median: 1,327,924 NOK
  - Max: 51,138,000,000 NOK

#### Tall 1340 (Salgsinntekt) - Used for driftsmargin
- **Zero values:** 359 (1.27%)
- **Negative values:** 39 (0.14%)
- **Near-zero (< 1,000 NOK):** 40 (0.14%)
- **Distribution:**
  - Min: -976,909 NOK
  - 1st percentile: 0 NOK
  - Median: 5,534,128 NOK
  - Max: 47,187,000,000 NOK

#### Total Assets (Tall 217 + Tall 194) - Used for gjeldsgrad
- **Zero values:** 400 (1.17%)
- **Negative values:** 48 (0.14%)
- **Near-zero (< 1,000 NOK):** 165 (0.48%)
- **Distribution:**
  - Min: -1,392,021 NOK
  - 1st percentile: 0 NOK
  - Median: 3,898,147 NOK
  - Max: 91,878,000,000 NOK

---

## Case Studies: Extreme Values

### Case 1: Likviditetsgrad = 6,967,346

**Company:** Orgnr 989255339, Year 2017
**Ratio calculation:** 6,967,346 NOK / 1 NOK = 6,967,346

**Raw data:**
- Omløpsmidler (Tall 194): 6,967,346 NOK
- Kortsiktig gjeld (Tall 85): **1 NOK**
- Total assets: 217,905,079 NOK
- Total debt: 1 NOK
- Sales: NaN

**Interpretation:** Company has 218M NOK in assets but only 1 NOK in short-term debt. This is likely a data entry error or a company in liquidation/restructuring. The ratio is technically correct but economically meaningless.

### Case 2: Gjeldsgrad = 12,702,030

**Company:** Orgnr 898206092, Year 2016
**Ratio calculation:** 12,702,030 NOK / 1 NOK = 12,702,030

**Raw data:**
- Total debt: 12,702,030 NOK (3.7M short-term + 9.0M long-term)
- Total assets: **1 NOK** (1 NOK anleggsmidler + 0 NOK omløpsmidler)
- Kortsiktig gjeld: 3,735,321 NOK
- Langsiktig gjeld: 8,966,709 NOK

**Interpretation:** Company has 12.7M NOK in debt but only 1 NOK in assets. This violates basic accounting principles (assets = liabilities + equity). Likely a data corruption or reporting error.

### Case 3: Driftsmargin = 1,530,192

**Company:** Orgnr 895113832, Year 2016
**Ratio calculation:** -1,530,192 NOK / -1 NOK = 1,530,192

**Raw data:**
- Driftsresultat (Tall 146): -1,530,192 NOK
- Salgsinntekt (Tall 1340): **-1 NOK**

**Interpretation:** Negative revenue of 1 NOK is nonsensical. The negative-divided-by-negative produces a large positive margin. This is a data error.

### Case 4: Negative Assets

**Company:** Orgnr 912855899, Year 2016
**Values:**
- Anleggsmidler (Tall 217): -90,042 NOK
- Omløpsmidler (Tall 194): -59,422 NOK
- Total assets: -149,464 NOK

**Interpretation:** Negative assets are impossible under standard accounting. This is either:
1. Data corruption in source system
2. Misclassification (liabilities reported as negative assets)
3. Database encoding error

---

## Impact on Analysis

### 1. Model Training Corruption
- Random Forest uses these extreme values to split nodes
- Creates spurious decision rules (e.g., "if likviditetsgrad > 100,000 then safe")
- Distorts feature importance rankings

### 2. Risk Stratification Errors
From `findings_report.md` (Sector C):

| Risk Tier | Gjeldsgrad | Likviditetsgrad | Driftsmargin |
|-----------|------------|-----------------|--------------|
| Low       | 1.30       | 6.65            | **-819%**    |
| Medium    | 0.95       | 7.12            | **76,289%**  |
| High      | **33.36**  | **-2,190**      | -93%         |

These profiles are uninterpretable due to extreme outliers pulling means.

### 3. Interaction Feature Contamination
Interaction features compound the problem:
```python
debt_liquidity_stress = total_gjeldsgrad / (likviditetsgrad_1 + 0.01)
```
When gjeldsgrad = 12M and likviditetsgrad = 6M:
- debt_liquidity_stress = 12M / 6M = 2 (looks normal!)
- But both inputs are corrupted

---

## Proposed Solutions

### Solution 1: Filter Near-Zero Denominators (RECOMMENDED)

Replace `safe_divide()` with a minimum threshold:

```python
def safe_divide(numerator, denominator, min_denominator=1000):
    """
    Divide but return NaN if denominator is too small, zero, or invalid.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        min_denominator: Minimum absolute value for valid division (default: 1000 NOK)

    Returns:
        Ratio or NaN if denominator is invalid
    """
    # Convert to absolute for threshold check
    abs_denom = np.abs(denominator)

    # Return NaN if denominator is NaN, zero, negative, or below threshold
    result = np.where(
        (pd.isna(denominator)) | (denominator == 0) | (denominator <= 0) | (abs_denom < min_denominator),
        np.nan,
        numerator / denominator
    )

    # Replace any remaining inf values
    result = pd.Series(result).replace([np.inf, -np.inf], np.nan)

    return result
```

**Rationale:**
- 1,000 NOK threshold: Below this, denominators are economically insignificant
- Filters 0.14-0.48% of observations per ratio (acceptable loss)
- Also rejects negative denominators (accounting errors)
- Median denominators are 1.3M-5.5M NOK, so 1,000 NOK is conservative

**Impact:**
- Likviditetsgrad: 158 → NaN (0.48% of Tall 85 values)
- Driftsmargin: 40 → NaN (0.14% of Tall 1340 values)
- Gjeldsgrad: 165 → NaN (0.48% of total assets)

### Solution 2: Winsorization (ALTERNATIVE)

Cap ratios at realistic bounds:

```python
def winsorize_ratio(ratio, lower_bound, upper_bound):
    """Clip ratio values to realistic economic ranges"""
    return np.clip(ratio, lower_bound, upper_bound)

# Apply after safe_divide
df['likviditetsgrad_1'] = winsorize_ratio(
    safe_divide(df['Tall 194'], df['Tall 85']),
    lower_bound=0,
    upper_bound=10
)
```

**Bounds (based on industry norms):**
- Likviditetsgrad: [0, 10]
- Gjeldsgrad: [0, 5]
- Driftsmargin: [-2, 2] (i.e., -200% to +200%)
- Egenkapitalandel: [-5, 1]
- Rentedekningsgrad: [-100, 100]
- Altman Z: [-10, 50]

**Rationale:**
- Preserves observations (no data loss)
- Caps at economically plausible values
- But: May mask data quality issues

**Downside:**
- Arbitrary threshold choices
- Companies with legitimate extreme values (rare) get misclassified

### Solution 3: Filter Negative Denominators

Reject observations with negative values in key balance sheet items:

```python
# Before calculating ratios
df = df[
    (df['Tall 85'] > 0) &  # Positive short-term debt
    (df['Tall 1340'] >= 0) &  # Non-negative revenue
    ((df['Tall 217'] + df['Tall 194']) > 0)  # Positive total assets
]
```

**Impact:** Filters 0.14-0.51% of observations per condition.

**Rationale:**
- Negative debt/assets violate accounting standards
- These are data errors, not economic realities
- Should be excluded regardless of ratio calculation

---

## Recommended Implementation Strategy

### Phase 1: Immediate Fix (Defensive)
1. **Add minimum denominator threshold** (1,000 NOK) to `safe_divide()`
2. **Filter negative denominators** (debt, assets, revenue < 0)
3. **Regenerate feature_dataset_v1.parquet**
4. **Re-run Sector C analysis** to verify fix

### Phase 2: Validation (Thorough)
5. **Winsorize as secondary check** (cap at 99th percentile per ratio)
6. **Compare results:** threshold vs winsorization vs original
7. **Document impact** on model performance (AUC, feature importance)

### Phase 3: Data Quality Audit (Long-term)
8. **Trace back to source:** Why do we have 1 NOK assets or -1 NOK revenue?
9. **Check regnskapstall_raw.parquet:** Are errors in raw data or feature engineering?
10. **Flag companies** with impossible values for exclusion

---

## Expected Results After Fix

### Observations Lost (Sector C)
- Near-zero denominators: ~0.5% per ratio
- Negative denominators: ~0.5% per ratio
- Total: ~1-2% of observations (acceptable for 34K dataset)

### Ratio Distributions (Expected)
- Likviditetsgrad: 0 to ~20 (vs current -3.6M to 6.9M)
- Gjeldsgrad: 0 to ~5 (vs current -3,271 to 12.7M)
- Driftsmargin: -5 to +5 (vs current -1.6M to 1.5M)
- Altman Z: -10 to +50 (vs current -3.6M to 591K)

### Risk Tier Profiles (Expected)
- All means within economically plausible ranges
- No margins of 76,289% or gjeldsgrad of 33.36
- Clearer separation between high-risk and low-risk tiers

### Model Performance (Expected)
- Similar or improved AUC (extreme values were noise, not signal)
- More interpretable feature importance (no spurious splits on outliers)
- Risk stratification profiles become economically coherent

---

## Code Changes Required

### File: `INF4090/data/features/build_features.py`

**Current code (lines 77-82):**
```python
def safe_divide(numerator, denominator):
    """Divide but return NaN if denominator is 0 or NaN"""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
```

**Proposed replacement:**
```python
def safe_divide(numerator, denominator, min_denominator=1000):
    """
    Divide but return NaN if denominator is too small, zero, negative, or invalid.

    Prevents extreme ratio values caused by near-zero or negative denominators.

    Args:
        numerator: Numerator value (can be negative, zero, or positive)
        denominator: Denominator value
        min_denominator: Minimum absolute value for valid division (default: 1000 NOK)
                        Values below this threshold are economically insignificant

    Returns:
        Ratio or NaN if denominator is invalid

    Examples:
        >>> safe_divide(1000, 500)  # Returns 2.0
        >>> safe_divide(1000, 1)     # Returns NaN (below threshold)
        >>> safe_divide(1000, -500)  # Returns NaN (negative)
        >>> safe_divide(1000, 0)     # Returns NaN (zero)
    """
    # Convert inputs to float64 to avoid overflow
    numerator = pd.Series(numerator, dtype='float64')
    denominator = pd.Series(denominator, dtype='float64')

    # Create mask for valid denominators:
    # - Not NaN
    # - Not zero
    # - Not negative (violates accounting standards)
    # - Absolute value >= min_denominator (economically significant)
    valid_mask = (
        (~pd.isna(denominator)) &
        (denominator != 0) &
        (denominator > 0) &  # Positive only
        (np.abs(denominator) >= min_denominator)
    )

    # Calculate result only for valid denominators
    result = pd.Series(index=numerator.index, dtype='float64')
    result[:] = np.nan  # Default to NaN
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Replace any remaining inf values (safety check)
    result = result.replace([np.inf, -np.inf], np.nan)

    return result
```

**Changes to ratio calculations (lines ~88-200):**

No changes needed - all calls to `safe_divide()` will automatically use the new validation logic:

```python
# These all automatically benefit from new validation
df_features['likviditetsgrad_1'] = safe_divide(df['Tall 194'], df['Tall 85'])
df_features['total_gjeldsgrad'] = safe_divide(df_features['_total_debt'], df_features['_total_assets'])
df_features['driftsmargin'] = safe_divide(df['Tall 146'], df['Tall 1340'])
# ... etc
```

**Optional: Add thresholds per ratio:**

```python
# If different thresholds needed per ratio
df_features['likviditetsgrad_1'] = safe_divide(df['Tall 194'], df['Tall 85'], min_denominator=1000)
df_features['driftsmargin'] = safe_divide(df['Tall 146'], df['Tall 1340'], min_denominator=100)  # Lower threshold for revenue
```

---

## Testing Plan

### Step 1: Unit Tests
```python
def test_safe_divide():
    # Normal case
    assert safe_divide(1000, 500) == 2.0

    # Near-zero (should be NaN)
    assert pd.isna(safe_divide(1000, 1))
    assert pd.isna(safe_divide(1000, 999))

    # Negative (should be NaN)
    assert pd.isna(safe_divide(1000, -500))

    # Zero (should be NaN)
    assert pd.isna(safe_divide(1000, 0))

    # Threshold boundary
    assert not pd.isna(safe_divide(1000, 1000))  # Exactly at threshold
    assert pd.isna(safe_divide(1000, 999))  # Just below threshold
```

### Step 2: Integration Test
1. Modify `build_features.py` with new `safe_divide()`
2. Run on small subset (100 companies)
3. Verify no extreme values (|ratio| > 100) remain
4. Check missing value rates (should increase by ~1-2%)

### Step 3: Full Regeneration
1. Delete old `feature_dataset_v1.parquet`
2. Run `python build_features.py` on full dataset
3. Verify row counts match (minus filtered observations)
4. Spot-check random companies for realistic ratios

### Step 4: Model Re-run
1. Re-run `sector_c_supervised_analysis.py`
2. Compare AUC (before vs after)
3. Compare feature importance rankings
4. Inspect risk tier profiles (should be economically coherent)
5. Update `findings_report.md` with cleaned results

---

## Conclusion

**Problem:** Extreme ratio values (6.9M likviditetsgrad, 12.7M gjeldsgrad) caused by near-zero and negative denominators.

**Root cause:** `safe_divide()` only handles exact zeros (inf), not near-zeros (1 NOK) or negatives (-1 NOK).

**Solution:** Add minimum denominator threshold (1,000 NOK) and reject negative values.

**Impact:** Filters 1-2% of observations with economically nonsensical data, improves model interpretability, and produces coherent risk profiles.

**Next steps:** Implement modified `safe_divide()` → regenerate features → re-run Sector C → validate results.
