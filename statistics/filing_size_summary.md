# Filing Behavior and Company Size Analysis - Summary

**Date:** 2025-12-01
**Total Companies Analyzed:** 114,848

---

## Key Findings

### 1. Filing Behavior

**Overall Filing Completeness:**
- **60.3%** (69,271) filed all 3 years (2016, 2017, 2018) ✅
- **39.7%** (45,577) filed partially (1-2 years only) ⚠️
- **0%** (0) never filed at all

**Most Common Filing Patterns:**
1. **2016+2017+2018** - 60.3% (filed all years)
2. **2017+2018** - 11.9% (missing 2016)
3. **2017 only** - 9.7% (only filed one year)
4. **2016+2017** - 8.1% (missing 2018)
5. **2018 only** - 5.6% (only filed 2018)

---

### 2. Company Size Distribution (by Employees)

| Size Category | Companies | Percentage | Avg Employees | Bankruptcy Rate |
|---------------|-----------|------------|---------------|-----------------|
| **Small (1-20)** | 28,988 | 25.2% | 10 | 4.10% |
| **Medium (21-100)** | 8,688 | 7.6% | 38 | 2.46% |
| **Large (100+)** | 1,286 | 1.1% | 378 | 1.48% |
| **Zero employees** | 46,452 | 40.5% | 0 | 7.27% |
| **Unknown** | 29,434 | 25.6% | N/A | 4.73% |

**Employee Data Coverage:** 74.4% of companies have employee data

---

### 3. Surprising Finding: Complete Filing Has HIGHER Bankruptcy Rate!

**Counterintuitive Result:**
- Companies that filed **ALL years**: **6.57%** bankruptcy rate
- Companies with **incomplete filing**: **3.60%** bankruptcy rate

**Why? Two explanations:**

1. **"Zero employees" companies distort the pattern:**
   - 46,452 companies (40.5%) have ZERO employees
   - These are likely shell companies, holding companies, or dormant entities
   - **Zero-employee companies with complete filing: 12.61% bankruptcy!**
   - This drags up the "complete filing" bankruptcy rate

2. **Companies file before going bankrupt:**
   - Distressed companies may file complete records to satisfy creditors
   - Once bankrupt, they stop filing (missing from later years)
   - Companies that skipped 2016 but filed 2017+2018 may have started operations recently

---

### 4. Size Matters: Larger = Safer

**Clear inverse relationship:**
- **Large (100+):** 1.48% bankruptcy rate
- **Medium (21-100):** 2.46% bankruptcy rate
- **Small (1-20):** 4.10% bankruptcy rate
- **Zero employees:** 7.27% bankruptcy rate

**Risk multiplier:** Small companies are **2.8x more likely** to go bankrupt than large companies

---

### 5. Highest Risk Profile

**Most Dangerous Combination:**
| Profile | Bankruptcy Rate |
|---------|-----------------|
| Zero employees + Complete filing | **12.61%** |
| Unknown size + Complete filing | 5.28% |
| Small + Incomplete filing | 4.67% |
| Small + Complete filing | 3.93% |

**Safest Combination:**
- Large companies (100+) with complete filing: **1.36%** bankruptcy rate

---

## Detailed Breakdowns

### Filing Patterns by Number of Years

| Years Filed | Companies | Bankruptcies | Bankruptcy Rate | Percentage |
|-------------|-----------|--------------|-----------------|------------|
| 1 year only | 22,538 | 549 | 2.44% | 19.6% |
| 2 years | 23,039 | 1,093 | 4.74% | 20.1% |
| 3 years (all) | 69,271 | 4,550 | 6.57% | 60.3% |

**Trend:** More years filed = higher bankruptcy rate (due to zero-employee effect)

---

### Filing Completeness by Company Size

| Size | Incomplete Filing | Complete Filing |
|------|-------------------|-----------------|
| **Large (100+)** | 8.4% | **91.6%** ✅ |
| **Medium (21-100)** | 14.7% | **85.3%** ✅ |
| **Small (1-20)** | 22.7% | **77.3%** ✅ |
| **Unknown** | 37.4% | 62.6% |
| **Zero employees** | **57.3%** | 42.7% ⚠️ |

**Observation:** Larger companies have much better filing compliance (91.6% for large vs 42.7% for zero-employee)

---

### Most Common Filing Patterns by Size

#### Small Companies (1-20 employees):
1. 2016+2017+2018: 77.3% (filed all years)
2. 2017+2018: 8.3%
3. 2016 only: 4.6%

#### Medium Companies (21-100 employees):
1. 2016+2017+2018: 85.3% (filed all years)
2. 2017+2018: 5.0%
3. 2016 only: 3.8%

#### Large Companies (100+ employees):
1. 2016+2017+2018: 91.6% (filed all years)
2. 2016 only: 2.9%
3. 2016+2017: 1.8%

**Trend:** Larger companies overwhelmingly file complete records

---

## Business Implications

### For Credit Risk Assessment

**High Risk Flags:**
1. **Zero employees** - 7.27% bankruptcy rate
2. **Unknown employee count** - suggests poor record-keeping
3. **Small companies (1-20)** - 4.10% bankruptcy rate
4. **Filed only 2018** - 8.47% bankruptcy rate (highest among filing patterns)

**Low Risk Signals:**
1. **Large companies (100+)** - 1.48% bankruptcy rate
2. **91.6% filing compliance** for large companies
3. **Medium-sized companies** also safer (2.46%)

### For Regulators

**Filing Compliance:**
- 60.3% file complete records (reasonable)
- 39.7% partial filing (concerning)
- Zero-employee companies least compliant (57.3% incomplete)

**Policy Recommendations:**
1. Enforce stricter filing requirements for zero-employee entities
2. Investigate companies that suddenly stop filing
3. Smaller companies need more support/guidance on filing requirements

### For the Bankruptcy Prediction Models

**Model Adjustments Needed:**
1. **Separate zero-employee companies:** They have unique risk profile (12.61% bankruptcy with complete filing)
2. **Size is protective:** Company size should be weighted heavily in models
3. **Filing patterns matter:** Companies filing only 2018 have 8.47% bankruptcy rate
4. **Non-linear relationship:** More filing ≠ less risk (due to zero-employee distortion)

---

## Comparison to Model Results

### Previous Finding (from unsupervised model):
- **"levert_alle_år"** (filed all years) was TOP predictor
- PC2 showed missingness indicators were highly predictive
- Companies with incomplete data had higher bankruptcy rates

### This Analysis Shows:
- **Opposite pattern for complete dataset:** Complete filers have 6.57% bankruptcy vs 3.60% for partial filers
- **Explanation:** Zero-employee companies (40.5% of dataset) create distortion
- **Refined insight:** Filing behavior is CONDITIONAL on company type
  - For operating companies (>0 employees): Incomplete filing = higher risk
  - For zero-employee entities: Complete filing = higher risk (paradoxically)

---

## Recommendations for Future Analysis

### 1. Segment Analysis
Run separate analyses for:
- **Operating companies** (>0 employees)
- **Zero-employee entities** (separate risk profile)
- **Unknown employee count** (missing data companies)

### 2. Refined Filing Metrics
Instead of simple "filed all years":
- **Consistency:** Did company file SAME set of fields each year?
- **Completeness:** % of required fields filled in
- **Timeliness:** Did they file on time or late?

### 3. Temporal Patterns
- **Companies that stop filing:** Track year they stopped (2016→2017→2018→bankruptcy)
- **New companies:** Those filing only 2017+2018 may be new (different risk)
- **Declining filing:** Companies filing fewer fields over time

### 4. Cross-Reference with Unsupervised Clusters
- Check if K-Means "distressed cluster" overlaps with:
  - Zero-employee companies
  - Small companies
  - Incomplete filers

---

## Files Generated

All files saved to: `INF4090/statistics/`

1. **company_filing_patterns.csv** (114,848 rows)
   - Each company's filing pattern across years
   - Columns: Orgnr, filed_2016, filed_2017, filed_2018, bankrupt, pattern, employees, size_category

2. **filing_pattern_statistics.csv**
   - Aggregate statistics for each filing pattern
   - Bankruptcy rates by pattern

3. **company_size_statistics.csv**
   - Statistics by company size category
   - Employee counts, bankruptcy rates

4. **size_filing_bankruptcy_analysis.csv**
   - Cross-tabulation: Size × Filing × Bankruptcy
   - Identifies highest/lowest risk combinations

5. **filing_size_summary.json**
   - Machine-readable summary statistics
   - For programmatic access

6. **filing_size_summary.md** (this file)
   - Human-readable analysis

---

## Conclusion

**Main Takeaways:**

1. **Company size is highly protective:** Large companies (100+) have 1.48% bankruptcy rate vs 7.27% for zero-employee entities

2. **Filing behavior is context-dependent:** Complete filing means different things for operating companies vs dormant entities

3. **Zero-employee companies are unique:** 40.5% of dataset, 12.61% bankruptcy rate when filing complete, distort aggregate statistics

4. **Data quality matters:** 74.4% have employee data; the 25.6% without are higher risk (4.73% bankruptcy)

5. **Model refinement needed:** Bankruptcy prediction models should segment by company size and employee status before analyzing filing behavior

**For your thesis:** This analysis reveals that **simple features can be misleading** without proper segmentation. The relationship between filing completeness and bankruptcy is NON-LINEAR and depends on company characteristics. This justifies the need for multiple machine learning approaches (supervised + unsupervised + segmented) rather than one-size-fits-all models.
