# Dataset Design Plan for Norwegian Bankruptcy Prediction

## Overview
We will create multiple datasets to support different analytical approaches and robustness checks.

---

## Dataset A: COMPLETE_CASES (Primary Analysis Dataset)

### Purpose
Main analysis with clean, complete 3-year time series for trend-based modeling.

### Inclusion Criteria
- Companies with accounting data for ALL 3 years (2016, 2017, 2018)
- Only companies that COULD have existed all 3 years (founded ≤2015)
- At least minimal accounting data in each year (not all fields empty)

### Expected Sample Size
- Bankrupt: ~4,550 companies
- Non-bankrupt: ~65,192 companies
- Total: ~69,742 companies
- Class balance: ~6.5% bankrupt

### Missing Data Handling
1. **Company-level**: Already handled by inclusion criteria (must have all 3 years)
2. **Field-level missing values** (empty strings in accounting columns):
   - **Forward fill** within company (2016→2017→2018)
   - Then **backward fill** within company (2018→2017)
   - Remaining nulls → **Industry median** by year
   - Create **missingness indicators**: e.g., `revenue_2018_was_missing` (0/1)

### Key Considerations
- **Survival bias**: Excludes companies that stopped filing (which may be informative)
- **Sample representativeness**: Are 3-year filers different from partial filers?
- **Temporal alignment**: All companies have same time window
- **Trend calculation**: Can reliably calculate growth rates, changes over time

### Output Files
1. `complete_cases_panel.parquet` - Long format (one row per company-year)
2. `complete_cases_wide.parquet` - Wide format (one row per company, all years as columns)
3. `complete_cases_metadata.parquet` - Company information (name, industry, etc.)

---

## Dataset B: MISSING_AS_FEATURE (Robustness Check #1)

### Purpose
Test whether missing data itself is predictive. More realistic scenario where you don't always have complete information.

### Inclusion Criteria
- ALL companies with ANY data in any year
- No exclusions based on data completeness

### Expected Sample Size
- Bankrupt: ~6,192 companies
- Non-bankrupt: ~111,041 companies
- Total: ~117,233 companies
- Class balance: ~5.3% bankrupt

### Missing Data Handling
1. **Missing years**:
   - If company has no data for a year → all fields NULL for that year
   - Create indicators: `has_data_2016`, `has_data_2017`, `has_data_2018`

2. **Missing fields within existing years**:
   - Keep as NULL initially
   - Create per-field indicators: `revenue_2018_missing`
   - Impute with: industry median or forward-fill (keep both versions)

3. **New derived features**:
   - `num_years_with_data` (0-3)
   - `filed_all_years` (0/1)
   - `filed_most_recent_year` (0/1) - 2018 filing status
   - `data_completeness_pct` (0-100)
   - `years_since_last_filing` (0-3)

### Key Considerations
- **Missing not at random (MNAR)**: Not filing is correlated with bankruptcy
- **Imputation bias**: Need to test multiple imputation strategies
- **Feature engineering**: Missingness patterns as predictors
- **Model selection**: Some models handle missing data better (XGBoost, LightGBM)
- **Comparison**: Can we improve prediction by using missingness as signal?

### Output Files
1. `missing_as_feature_panel.parquet` - Long format with NULL values preserved
2. `missing_as_feature_wide.parquet` - Wide format with missingness indicators
3. `missing_as_feature_imputed.parquet` - Version with imputed values
4. `missing_as_feature_metadata.parquet` - Company info + filing patterns

---

## Dataset C: CONSECUTIVE_YEARS (Robustness Check #2)

### Purpose
Middle ground - includes more companies than A, less complex than B. Good for sensitivity analysis.

### Inclusion Criteria
- Companies with at least 2 CONSECUTIVE years of data:
  - 2016 AND 2017, OR
  - 2017 AND 2018
- Prefer companies with 2017+2018 (most recent, closer to bankruptcy event)

### Expected Sample Size
- Larger than Dataset A, smaller than Dataset B
- Estimate: ~80,000-90,000 companies

### Missing Data Handling
1. **For 3-year companies**: Same as Dataset A
2. **For 2-year companies**:
   - Missing year → NULL
   - Can calculate 1 year-over-year change (not 2)
   - Flag: `has_3_years` vs `has_2_years`

### Key Considerations
- **Heterogeneous time coverage**: Some have 3 years, some 2
- **Trend features**: Can calculate for 2-year pairs only
- **Model complexity**: Need to handle mixed temporal coverage
- **Use case**: Test if model is robust to some missing data

### Output Files
1. `consecutive_years_panel.parquet`
2. `consecutive_years_wide.parquet`
3. `consecutive_years_metadata.parquet`

---

## Dataset D: EARLY_STAGE (Supplementary Analysis)

### Purpose
Separate analysis for companies founded 2016-2018 (startups/new companies). Different research question.

### Inclusion Criteria
- Companies founded in 2016, 2017, or 2018
- Have data for at least their founding year

### Expected Sample Size
- Estimate: ~30,000-45,000 companies
- Higher bankruptcy rate expected (startup failure)

### Missing Data Handling
- Only use years AFTER founding
- No forward-fill from before founding (doesn't exist)
- Industry median imputation for same-aged companies

### Key Considerations
- **Different risk profile**: Startup failure vs established company bankruptcy
- **Limited history**: Only 1-2 years of data
- **Separate model**: Should NOT mix with established companies
- **Research value**: "Can we predict early-stage failure?"
- **Practical use**: Banks evaluating startup loan applications

### Output Files
1. `early_stage_panel.parquet`
2. `early_stage_metadata.parquet`

---

## Common Elements Across All Datasets

### Standard Fields (All Datasets)

**Identifiers:**
- `orgnr` - Organization number (cleaned, no spaces/dashes)
- `year` - Accounting year (2016, 2017, 2018)
- `bankrupt` - Target variable (1=bankrupt in 2019, 0=survived)

**Accounting Metrics (Raw):**
- `revenue` (Tall 1340 - Salgsinntekt)
- `other_operating_income` (Tall 7709)
- `total_income` (Tall 72)
- `fixed_assets` (Tall 217)
- `current_assets` (Tall 194)
- `long_term_debt` (Tall 86)
- `short_term_debt` (Tall 85)
- `operating_result` (Tall 146 - EBIT)
- `financial_expenses` (Tall 17130)

**Company Information:**
- `company_name`
- `industry_code` (Næringskode)
- `industry_description`
- `sector_code`
- `sector_description`
- `org_form` (AS, ASA, etc.)
- `municipality`
- `county`
- `num_employees_category`
- `founded_date`

**Data Quality Flags:**
- `data_source` (which original file)
- `has_complete_data` (all fields present)
- `imputed_fields` (list of which fields were imputed)

### Derived Financial Ratios (Calculated for All)

**Liquidity Ratios:**
- `current_ratio` = current_assets / short_term_debt
- `quick_ratio` = (current_assets - inventory) / short_term_debt

**Leverage Ratios:**
- `debt_ratio` = (long_term_debt + short_term_debt) / (fixed_assets + current_assets)
- `debt_to_equity` = total_debt / equity

**Profitability Ratios:**
- `operating_margin` = operating_result / total_income
- `roa` = operating_result / total_assets
- `roe` = net_income / equity

**Efficiency:**
- `asset_turnover` = revenue / total_assets

**Bankruptcy Prediction:**
- `altman_z_score` (if we have all needed components)

### Trend Features (Where Applicable)

**Year-over-Year Changes:**
- `revenue_growth_1617` = (revenue_2017 - revenue_2016) / revenue_2016
- `revenue_growth_1718` = (revenue_2018 - revenue_2017) / revenue_2017
- `debt_change_1617`, `debt_change_1718`
- `operating_result_change_1617`, `operating_result_change_1718`

**Multi-year Trends:**
- `revenue_cagr_1618` = compound annual growth rate
- `revenue_volatility` = std deviation of revenues
- `consistent_profitability` = (all years profitable: 0/1)
- `deteriorating_liquidity` = (current_ratio trending down: 0/1)

---

## Key Decisions to Make

### 1. Imputation Strategy
**Options:**
- A) Forward-fill → Industry median
- B) KNN imputation (similar companies)
- C) Model-based (predict missing values)
- D) Multiple imputation (create multiple datasets)

**Recommendation:** Start with (A), test sensitivity with (B) and (C)

### 2. Industry Grouping
- Use which level of industry code? (2-digit, 3-digit, 5-digit?)
- How many companies minimum per industry for median calculation?
- Group rare industries into "Other"?

**Recommendation:** 2-digit for median imputation (broader groups, more stable)

### 3. Handling Extreme Outliers
- Winsorization (cap at 1st/99th percentile)?
- Log transformation?
- Flag extreme values?

**Recommendation:** Flag outliers, winsorize for ratios, keep raw values

### 4. Company Founding Date
- Where to get this? (Stiftelsesdato field)
- If missing, how to handle?
- If founded DURING a year (e.g., June 2017), include or exclude?

**Recommendation:** Use Stiftelsesdato, founded ≤2015-12-31 for Dataset A

### 5. Bankrupt Company Definition
- Use bankruptcy status from which source?
- What if company appears in konkurser2019 but also in books files?

**Recommendation:** Any company in konkurser2019 = bankrupt (1), rest = (0)

### 6. File Format
- CSV (human readable, universal)
- Parquet (efficient, preserves types, compressed)
- Both?

**Recommendation:** Parquet for main files, CSV for metadata (easy inspection)

---

## Data Validation Checks

For each dataset, implement:

1. **Uniqueness**: No duplicate (orgnr, year) pairs in panel data
2. **Temporal consistency**: No data from future years
3. **Logical constraints**:
   - All financial values ≥ 0 (or flag negatives if valid)
   - Debt ratio between 0-1 (or > 1 if overleveraged)
4. **Missingness**: Document % missing for each field
5. **Outliers**: Flag values > 3 SD from mean
6. **Class balance**: Report bankrupt/non-bankrupt ratio
7. **Industry distribution**: Check if representative

---

## Output Directory Structure

```
data/
├── raw/                          # Original Excel files
│   ├── books2016.xlsx
│   ├── book2017.xlsx
│   ├── books2018.xlsx
│   └── konkurser2019.xlsx
│
├── processed/
│   ├── dataset_a_complete_cases/
│   │   ├── panel.parquet
│   │   ├── wide.parquet
│   │   ├── metadata.parquet
│   │   └── data_report.txt
│   │
│   ├── dataset_b_missing_as_feature/
│   │   ├── panel.parquet
│   │   ├── panel_imputed.parquet
│   │   ├── wide.parquet
│   │   ├── metadata.parquet
│   │   └── data_report.txt
│   │
│   ├── dataset_c_consecutive_years/
│   │   └── ...
│   │
│   └── dataset_d_early_stage/
│       └── ...
│
└── analysis/
    ├── exploratory_data_analysis.ipynb
    ├── model_training.ipynb
    └── results/
```

---

## Timeline / Priority

1. **PHASE 1 (First)**: Dataset A - Complete Cases
   - Cleanest, most straightforward
   - Start analysis immediately
   - Primary results for thesis

2. **PHASE 2 (Second)**: Dataset B - Missing as Feature
   - Robustness check
   - Test if missingness improves prediction

3. **PHASE 3 (Third)**: Dataset C - Consecutive Years
   - Additional robustness
   - Optional if time permits

4. **PHASE 4 (Optional)**: Dataset D - Early Stage
   - Separate research question
   - Mention in discussion/future work if no time

---

## Next Steps

1. Verify field mappings (Tall codes → meaningful names)
2. Decide on imputation strategy
3. Decide on industry grouping level
4. Build Dataset A first
5. Generate data quality report
6. Begin EDA on Dataset A
