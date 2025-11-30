"""
Feature Engineering for Norwegian Bankruptcy Prediction
========================================================

This script calculates 45 derived features from raw accounting data.
All features are based on established bankruptcy prediction theory:
- Beaver (1966): Financial ratios as predictors of failure
- Altman (1968): Z-Score and discriminant analysis
- Ohlson (1980): Logistic regression bankruptcy model

Input: data/processed/norwegian_companies_panel.parquet
Output: data/features/feature_dataset_v1.parquet

All feature names are in Norwegian as per project requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("FEATURE ENGINEERING - NORWEGIAN BANKRUPTCY PREDICTION")
print("=" * 80)
print(f"Started: {datetime.now()}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading raw panel data...")

# Determine correct path based on where we're running from
if Path('../../INF4090/data/processed/norwegian_companies_panel.parquet').exists():
    input_file = Path('../../INF4090/data/processed/norwegian_companies_panel.parquet')
elif Path('../processed/norwegian_companies_panel.parquet').exists():
    input_file = Path('../processed/norwegian_companies_panel.parquet')
else:
    # Assume we're in INF4090 structure
    input_file = Path('C:/Users/magnu/Desktop/AI Management/INF4090/data/processed/norwegian_companies_panel.parquet')

print(f"  Reading from: {input_file}")
df = pd.read_parquet(input_file)

print(f"  Loaded: {len(df):,} rows")
print(f"  Companies: {df['Orgnr'].nunique():,}")
print(f"  Years: {sorted(df['year'].unique())}")

# Create a copy for feature engineering
df_features = df.copy()

# Track what we add for documentation
features_added = []
calculation_log = []

def log_feature(name, formula, theory, calculated_count, missing_count, stats):
    """Track each feature calculation for documentation"""
    features_added.append(name)
    calculation_log.append({
        'name': name,
        'formula': formula,
        'theory': theory,
        'calculated': calculated_count,
        'missing': missing_count,
        'mean': stats.get('mean'),
        'median': stats.get('median'),
        'std': stats.get('std'),
        'min': stats.get('min'),
        'max': stats.get('max')
    })

# ============================================================================
# CATEGORY 1: FINANCIAL RATIOS (11 features)
# ============================================================================
print("\n[2/6] Calculating Financial Ratios...")
print("  Theoretical basis: Beaver (1966), Altman (1968), Ohlson (1980)")

# Helper function to safely divide (avoid division by zero)
def safe_divide(numerator, denominator):
    """Divide but return NaN if denominator is 0 or NaN"""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

# 1.1 LIQUIDITY RATIOS
print("\n  [1.1] Liquidity Ratios...")

# Current Ratio (Likviditetsgrad 1)
df_features['likviditetsgrad_1'] = safe_divide(df['Tall 194'], df['Tall 85'])
calc_count = df_features['likviditetsgrad_1'].notna().sum()
miss_count = df_features['likviditetsgrad_1'].isna().sum()
stats = df_features['likviditetsgrad_1'].describe().to_dict()
log_feature('likviditetsgrad_1', 'Tall 194 / Tall 85',
            'Beaver (1966), Altman (1968)', calc_count, miss_count, stats)
print(f"    likviditetsgrad_1: {calc_count:,} calculated, {miss_count:,} missing")

# Quick Ratio (Likviditetsgrad 2) - same as current ratio since we don't have inventory
df_features['likviditetsgrad_2'] = df_features['likviditetsgrad_1'].copy()
log_feature('likviditetsgrad_2', 'Tall 194 / Tall 85 (no inventory data)',
            'Ohlson (1980)', calc_count, miss_count, stats)
print(f"    likviditetsgrad_2: {calc_count:,} calculated (same as likviditetsgrad_1)")

# 1.2 LEVERAGE RATIOS
print("\n  [1.2] Leverage Ratios...")

# Calculate total assets and total debt first (used in multiple ratios)
df_features['_total_assets'] = df['Tall 217'] + df['Tall 194']  # Temporary column
df_features['_total_debt'] = df['Tall 86'] + df['Tall 85']  # Temporary column

# Total Debt Ratio (Total Gjeldsgrad)
df_features['total_gjeldsgrad'] = safe_divide(df_features['_total_debt'], df_features['_total_assets'])
calc_count = df_features['total_gjeldsgrad'].notna().sum()
miss_count = df_features['total_gjeldsgrad'].isna().sum()
stats = df_features['total_gjeldsgrad'].describe().to_dict()
log_feature('total_gjeldsgrad', '(Tall 86 + Tall 85) / (Tall 217 + Tall 194)',
            'Altman (1968), Zmijewski (1984)', calc_count, miss_count, stats)
print(f"    total_gjeldsgrad: {calc_count:,} calculated, {miss_count:,} missing")

# Long-term Debt Ratio (Langsiktig Gjeldsgrad)
df_features['langsiktig_gjeldsgrad'] = safe_divide(df['Tall 86'], df_features['_total_assets'])
calc_count = df_features['langsiktig_gjeldsgrad'].notna().sum()
miss_count = df_features['langsiktig_gjeldsgrad'].isna().sum()
stats = df_features['langsiktig_gjeldsgrad'].describe().to_dict()
log_feature('langsiktig_gjeldsgrad', 'Tall 86 / (Tall 217 + Tall 194)',
            'Leverage analysis', calc_count, miss_count, stats)
print(f"    langsiktig_gjeldsgrad: {calc_count:,} calculated, {miss_count:,} missing")

# Short-term Debt Ratio (Kortsiktig Gjeldsgrad)
df_features['kortsiktig_gjeldsgrad'] = safe_divide(df['Tall 85'], df_features['_total_assets'])
calc_count = df_features['kortsiktig_gjeldsgrad'].notna().sum()
miss_count = df_features['kortsiktig_gjeldsgrad'].isna().sum()
stats = df_features['kortsiktig_gjeldsgrad'].describe().to_dict()
log_feature('kortsiktig_gjeldsgrad', 'Tall 85 / (Tall 217 + Tall 194)',
            'Short-term liquidity pressure', calc_count, miss_count, stats)
print(f"    kortsiktig_gjeldsgrad: {calc_count:,} calculated, {miss_count:,} missing")

# Equity Ratio (Egenkapitalandel)
df_features['egenkapitalandel'] = 1 - df_features['total_gjeldsgrad']
calc_count = df_features['egenkapitalandel'].notna().sum()
miss_count = df_features['egenkapitalandel'].isna().sum()
stats = df_features['egenkapitalandel'].describe().to_dict()
log_feature('egenkapitalandel', '1 - total_gjeldsgrad',
            'Norwegian accounting standards', calc_count, miss_count, stats)
print(f"    egenkapitalandel: {calc_count:,} calculated, {miss_count:,} missing")

# 1.3 PROFITABILITY RATIOS
print("\n  [1.3] Profitability Ratios...")

# Operating Margin (Driftsmargin)
df_features['driftsmargin'] = safe_divide(df['Tall 146'], df['Tall 72'])
calc_count = df_features['driftsmargin'].notna().sum()
miss_count = df_features['driftsmargin'].isna().sum()
stats = df_features['driftsmargin'].describe().to_dict()
log_feature('driftsmargin', 'Tall 146 / Tall 72',
            'Taffler (1983)', calc_count, miss_count, stats)
print(f"    driftsmargin: {calc_count:,} calculated, {miss_count:,} missing")

# Return on Assets (Totalkapitalrentabilitet)
df_features['totalkapitalrentabilitet'] = safe_divide(df['Tall 146'], df_features['_total_assets'])
calc_count = df_features['totalkapitalrentabilitet'].notna().sum()
miss_count = df_features['totalkapitalrentabilitet'].isna().sum()
stats = df_features['totalkapitalrentabilitet'].describe().to_dict()
log_feature('totalkapitalrentabilitet', 'Tall 146 / (Tall 217 + Tall 194)',
            'Altman (1968), Beaver (1966)', calc_count, miss_count, stats)
print(f"    totalkapitalrentabilitet: {calc_count:,} calculated, {miss_count:,} missing")

# Asset Turnover (Omsetningsgrad)
df_features['omsetningsgrad'] = safe_divide(df['Tall 1340'], df_features['_total_assets'])
calc_count = df_features['omsetningsgrad'].notna().sum()
miss_count = df_features['omsetningsgrad'].isna().sum()
stats = df_features['omsetningsgrad'].describe().to_dict()
log_feature('omsetningsgrad', 'Tall 1340 / (Tall 217 + Tall 194)',
            'Asset efficiency', calc_count, miss_count, stats)
print(f"    omsetningsgrad: {calc_count:,} calculated, {miss_count:,} missing")

# 1.4 COVERAGE RATIOS
print("\n  [1.4] Coverage Ratios...")

# Interest Coverage (Rentedekningsgrad)
df_features['rentedekningsgrad'] = safe_divide(df['Tall 146'], df['Tall 17130'])
calc_count = df_features['rentedekningsgrad'].notna().sum()
miss_count = df_features['rentedekningsgrad'].isna().sum()
stats = df_features['rentedekningsgrad'].describe().to_dict()
log_feature('rentedekningsgrad', 'Tall 146 / Tall 17130',
            'Times Interest Earned ratio', calc_count, miss_count, stats)
print(f"    rentedekningsgrad: {calc_count:,} calculated, {miss_count:,} missing")

# 1.5 COMPOSITE SCORES
print("\n  [1.5] Composite Scores...")

# Altman Z-Score (Simplified for private companies)
# Z = 0.717*X1 + 3.107*X3 + 0.420*X4 + 0.998*X5
# Note: X2 (retained earnings) not available in our data

working_capital = df['Tall 194'] - df['Tall 85']
equity = df_features['_total_assets'] - df_features['_total_debt']

X1 = safe_divide(working_capital, df_features['_total_assets'])
X3 = safe_divide(df['Tall 146'], df_features['_total_assets'])
X4 = safe_divide(equity, df_features['_total_debt'])
X5 = safe_divide(df['Tall 1340'], df_features['_total_assets'])

df_features['altman_z_score'] = (0.717 * X1 + 3.107 * X3 + 0.420 * X4 + 0.998 * X5)
calc_count = df_features['altman_z_score'].notna().sum()
miss_count = df_features['altman_z_score'].isna().sum()
stats = df_features['altman_z_score'].describe().to_dict()
log_feature('altman_z_score', '0.717*X1 + 3.107*X3 + 0.420*X4 + 0.998*X5 (simplified)',
            'Altman (1968, revised 1983)', calc_count, miss_count, stats)
print(f"    altman_z_score: {calc_count:,} calculated, {miss_count:,} missing")

print(f"\n  Financial Ratios: 11 features calculated")

# ============================================================================
# CATEGORY 2: TEMPORAL FEATURES (10 features)
# ============================================================================
print("\n[3/6] Calculating Temporal Features...")
print("  Note: Only calculated for companies with data in consecutive years")

# Reshape data to have one row per company with columns for each year
df_pivot = df_features.pivot_table(
    index='Orgnr',
    columns='year',
    values=['Tall 1340', '_total_assets', '_total_debt', 'likviditetsgrad_1',
            'total_gjeldsgrad', 'Tall 146'],
    aggfunc='first'
)

# Flatten column names
df_pivot.columns = [f'{col[0]}_{ col[1]}' for col in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# 2.1 GROWTH RATES
print("\n  [2.1] Growth Rates...")

# Revenue Growth
df_pivot['omsetningsvekst_1617'] = safe_divide(
    df_pivot['Tall 1340_2017'] - df_pivot['Tall 1340_2016'],
    df_pivot['Tall 1340_2016']
)
df_pivot['omsetningsvekst_1718'] = safe_divide(
    df_pivot['Tall 1340_2018'] - df_pivot['Tall 1340_2017'],
    df_pivot['Tall 1340_2017']
)

# Asset Growth
df_pivot['aktivavekst_1617'] = safe_divide(
    df_pivot['_total_assets_2017'] - df_pivot['_total_assets_2016'],
    df_pivot['_total_assets_2016']
)
df_pivot['aktivavekst_1718'] = safe_divide(
    df_pivot['_total_assets_2018'] - df_pivot['_total_assets_2017'],
    df_pivot['_total_assets_2017']
)

# Debt Growth
df_pivot['gjeldsvekst_1617'] = safe_divide(
    df_pivot['_total_debt_2017'] - df_pivot['_total_debt_2016'],
    df_pivot['_total_debt_2016']
)
df_pivot['gjeldsvekst_1718'] = safe_divide(
    df_pivot['_total_debt_2018'] - df_pivot['_total_debt_2017'],
    df_pivot['_total_debt_2017']
)

# Merge growth features back to main dataset
growth_cols = ['Orgnr', 'omsetningsvekst_1617', 'omsetningsvekst_1718',
               'aktivavekst_1617', 'aktivavekst_1718',
               'gjeldsvekst_1617', 'gjeldsvekst_1718']
df_features = df_features.merge(df_pivot[growth_cols], on='Orgnr', how='left')

for col in growth_cols[1:]:  # Skip Orgnr
    calc_count = df_features[col].notna().sum()
    miss_count = df_features[col].isna().sum()
    stats = df_features[col].describe().to_dict()
    log_feature(col, f'Year-over-year change',
                'Temporal dynamics', calc_count, miss_count, stats)
    print(f"    {col}: {calc_count:,} calculated, {miss_count:,} missing")

# 2.2 TREND INDICATORS
print("\n  [2.2] Trend Indicators...")

# Deteriorating Liquidity
df_pivot['fallende_likviditet'] = (
    (df_pivot['likviditetsgrad_1_2018'] < df_pivot['likviditetsgrad_1_2017']) &
    (df_pivot['likviditetsgrad_1_2017'] < df_pivot['likviditetsgrad_1_2016'])
).astype(float)

# Consistent Losses
df_pivot['konsistent_underskudd'] = (
    (df_pivot['Tall 146_2016'] < 0) &
    (df_pivot['Tall 146_2017'] < 0) &
    (df_pivot['Tall 146_2018'] < 0)
).astype(float)

# Increasing Leverage
df_pivot['økende_gjeldsgrad'] = (
    (df_pivot['total_gjeldsgrad_2018'] > df_pivot['total_gjeldsgrad_2017']) &
    (df_pivot['total_gjeldsgrad_2017'] > df_pivot['total_gjeldsgrad_2016'])
).astype(float)

# Merge trend features
trend_cols = ['Orgnr', 'fallende_likviditet', 'konsistent_underskudd', 'økende_gjeldsgrad']
df_features = df_features.merge(df_pivot[trend_cols], on='Orgnr', how='left')

for col in trend_cols[1:]:
    calc_count = df_features[col].notna().sum()
    miss_count = df_features[col].isna().sum()
    stats = {'mean': df_features[col].mean(), 'sum': df_features[col].sum()}
    log_feature(col, f'3-year trend indicator (0/1)',
                'Consistent deterioration signal', calc_count, miss_count, stats)
    print(f"    {col}: {df_features[col].sum():,.0f} companies with this pattern")

# 2.3 VOLATILITY
print("\n  [2.3] Volatility Measures...")

# Revenue Volatility (coefficient of variation)
revenue_cols = ['Tall 1340_2016', 'Tall 1340_2017', 'Tall 1340_2018']
df_pivot['omsetningsvolatilitet'] = df_pivot[revenue_cols].std(axis=1) / df_pivot[revenue_cols].mean(axis=1)

df_features = df_features.merge(df_pivot[['Orgnr', 'omsetningsvolatilitet']], on='Orgnr', how='left')

calc_count = df_features['omsetningsvolatilitet'].notna().sum()
miss_count = df_features['omsetningsvolatilitet'].isna().sum()
stats = df_features['omsetningsvolatilitet'].describe().to_dict()
log_feature('omsetningsvolatilitet', 'std(revenues) / mean(revenues)',
            'Business stability indicator', calc_count, miss_count, stats)
print(f"    omsetningsvolatilitet: {calc_count:,} calculated, {miss_count:,} missing")

print(f"\n  Temporal Features: 10 features calculated")

# ============================================================================
# CATEGORY 3: MISSINGNESS FEATURES (7 features)
# ============================================================================
print("\n[4/6] Calculating Missingness Features...")
print("  Empirical basis: 76.5% of bankrupt companies didn't file 2018 data")

# Group by company to calculate filing patterns
company_filing = df_features.groupby('Orgnr').agg({
    'year': lambda x: list(x),
    'Tall 1340': lambda x: x.notna().sum(),  # Count of non-missing revenue entries
}).reset_index()

company_filing.columns = ['Orgnr', 'years_list', 'revenue_entries']

# Filed All Years
company_filing['levert_alle_år'] = company_filing['years_list'].apply(
    lambda x: 1 if len(x) == 3 and set(x) == {2016, 2017, 2018} else 0
)

# Filed 2018
company_filing['levert_2018'] = company_filing['years_list'].apply(
    lambda x: 1 if 2018 in x else 0
)

# Number of Years Filed
company_filing['antall_år_levert'] = company_filing['years_list'].apply(len)

# Merge filing features
filing_cols = ['Orgnr', 'levert_alle_år', 'levert_2018', 'antall_år_levert']
df_features = df_features.merge(company_filing[filing_cols], on='Orgnr', how='left')

for col in filing_cols[1:]:
    calc_count = len(df_features)  # All rows get this feature
    stats = df_features.groupby('Orgnr')[col].first().describe().to_dict()
    log_feature(col, 'Filing pattern indicator',
                'Non-filing as bankruptcy predictor', calc_count, 0, stats)
    print(f"    {col}: All companies ({calc_count:,} rows)")

# Accounting Completeness (all 9 Tall fields present)
required_fields = ['Tall 1340', 'Tall 7709', 'Tall 72', 'Tall 217', 'Tall 194',
                   'Tall 86', 'Tall 85', 'Tall 146', 'Tall 17130']

df_features['regnskapskomplett'] = df_features[required_fields].notna().all(axis=1).astype(int)

calc_count = len(df_features)
stats = df_features['regnskapskomplett'].describe().to_dict()
log_feature('regnskapskomplett', 'All 9 Tall fields present (0/1)',
            'Data quality indicator', calc_count, 0, stats)
print(f"    regnskapskomplett: {df_features['regnskapskomplett'].sum():,} complete records")

# Cannot Calculate Key Ratios
df_features['kan_ikke_beregne_likviditet'] = (
    df_features['Tall 194'].isna() | df_features['Tall 85'].isna()
).astype(int)

df_features['kan_ikke_beregne_gjeldsgrad'] = (
    df_features['_total_assets'].isna() | df_features['_total_debt'].isna()
).astype(int)

for col in ['kan_ikke_beregne_likviditet', 'kan_ikke_beregne_gjeldsgrad']:
    calc_count = len(df_features)
    stats = df_features[col].describe().to_dict()
    log_feature(col, 'Missing data for ratio calculation (0/1)',
                'Missingness indicator', calc_count, 0, stats)
    print(f"    {col}: {df_features[col].sum():,} cannot calculate")

print(f"\n  Missingness Features: 7 features calculated")

# ============================================================================
# CATEGORY 4: COMPANY CHARACTERISTICS (4 features)
# ============================================================================
print("\n[5/6] Calculating Company Characteristics...")

# Company Age
df_features['Stiftelsesdato_clean'] = pd.to_datetime(df_features['Stiftelsesdato'], errors='coerce')
df_features['selskapsalder'] = df_features['year'] - df_features['Stiftelsesdato_clean'].dt.year

calc_count = df_features['selskapsalder'].notna().sum()
miss_count = df_features['selskapsalder'].isna().sum()
stats = df_features['selskapsalder'].describe().to_dict()
log_feature('selskapsalder', 'year - year(Stiftelsesdato)',
            'Liability of newness (Stinchcombe 1965)', calc_count, miss_count, stats)
print(f"    selskapsalder: {calc_count:,} calculated, {miss_count:,} missing")

# Young Company Indicator
df_features['nytt_selskap'] = (df_features['selskapsalder'] <= 5).astype(float)
df_features.loc[df_features['selskapsalder'].isna(), 'nytt_selskap'] = np.nan

calc_count = df_features['nytt_selskap'].notna().sum()
miss_count = df_features['nytt_selskap'].isna().sum()
stats = {'sum': df_features['nytt_selskap'].sum(), 'mean': df_features['nytt_selskap'].mean()}
log_feature('nytt_selskap', 'selskapsalder <= 5',
            'Young company risk', calc_count, miss_count, stats)
print(f"    nytt_selskap: {df_features['nytt_selskap'].sum():,.0f} young companies")

# Log Total Assets
df_features['log_totalkapital'] = np.log(df_features['_total_assets'] + 1)

calc_count = df_features['log_totalkapital'].notna().sum()
miss_count = df_features['log_totalkapital'].isna().sum()
stats = df_features['log_totalkapital'].describe().to_dict()
log_feature('log_totalkapital', 'log(Tall 217 + Tall 194 + 1)',
            'Company size proxy', calc_count, miss_count, stats)
print(f"    log_totalkapital: {calc_count:,} calculated, {miss_count:,} missing")

# Log Revenue
df_features['log_omsetning'] = np.log(df_features['Tall 1340'] + 1)

calc_count = df_features['log_omsetning'].notna().sum()
miss_count = df_features['log_omsetning'].isna().sum()
stats = df_features['log_omsetning'].describe().to_dict()
log_feature('log_omsetning', 'log(Tall 1340 + 1)',
            'Revenue size', calc_count, miss_count, stats)
print(f"    log_omsetning: {calc_count:,} calculated, {miss_count:,} missing")

print(f"\n  Company Characteristics: 4 features calculated")

# ============================================================================
# CATEGORY 5: WARNING SIGNALS (5 features)
# ============================================================================
print("\n[6/6] Calculating Warning Signals...")

# Negative Equity
df_features['negativ_egenkapital'] = (df_features['egenkapitalandel'] < 0).astype(float)
df_features.loc[df_features['egenkapitalandel'].isna(), 'negativ_egenkapital'] = np.nan

calc_count = df_features['negativ_egenkapital'].notna().sum()
stats = {'sum': df_features['negativ_egenkapital'].sum()}
log_feature('negativ_egenkapital', 'egenkapitalandel < 0',
            'Technical insolvency', calc_count, 0, stats)
print(f"    negativ_egenkapital: {df_features['negativ_egenkapital'].sum():,.0f} companies")

# Overleveraged
df_features['sterkt_overbelånt'] = (df_features['total_gjeldsgrad'] > 0.8).astype(float)
df_features.loc[df_features['total_gjeldsgrad'].isna(), 'sterkt_overbelånt'] = np.nan

calc_count = df_features['sterkt_overbelånt'].notna().sum()
stats = {'sum': df_features['sterkt_overbelånt'].sum()}
log_feature('sterkt_overbelånt', 'total_gjeldsgrad > 0.8',
            'High leverage warning', calc_count, 0, stats)
print(f"    sterkt_overbelånt: {df_features['sterkt_overbelånt'].sum():,.0f} companies")

# Cannot Cover Interest
df_features['kan_ikke_dekke_renter'] = (df_features['rentedekningsgrad'] < 1.0).astype(float)
df_features.loc[df_features['rentedekningsgrad'].isna(), 'kan_ikke_dekke_renter'] = np.nan

calc_count = df_features['kan_ikke_dekke_renter'].notna().sum()
stats = {'sum': df_features['kan_ikke_dekke_renter'].sum()}
log_feature('kan_ikke_dekke_renter', 'rentedekningsgrad < 1.0',
            'Interest coverage failure', calc_count, 0, stats)
print(f"    kan_ikke_dekke_renter: {df_features['kan_ikke_dekke_renter'].sum():,.0f} companies")

# Low Liquidity
df_features['lav_likviditet'] = (df_features['likviditetsgrad_1'] < 1.0).astype(float)
df_features.loc[df_features['likviditetsgrad_1'].isna(), 'lav_likviditet'] = np.nan

calc_count = df_features['lav_likviditet'].notna().sum()
stats = {'sum': df_features['lav_likviditet'].sum()}
log_feature('lav_likviditet', 'likviditetsgrad_1 < 1.0',
            'Liquidity crisis', calc_count, 0, stats)
print(f"    lav_likviditet: {df_features['lav_likviditet'].sum():,.0f} companies")

# Operating Loss
df_features['driftsunderskudd'] = (df_features['Tall 146'] < 0).astype(float)
df_features.loc[df_features['Tall 146'].isna(), 'driftsunderskudd'] = np.nan

calc_count = df_features['driftsunderskudd'].notna().sum()
stats = {'sum': df_features['driftsunderskudd'].sum()}
log_feature('driftsunderskudd', 'Tall 146 < 0',
            'Unprofitable operations', calc_count, 0, stats)
print(f"    driftsunderskudd: {df_features['driftsunderskudd'].sum():,.0f} companies")

print(f"\n  Warning Signals: 5 features calculated")

# ============================================================================
# CLEANUP AND SAVE
# ============================================================================
print("\n" + "=" * 80)
print("FINALIZING DATASET")
print("=" * 80)

# Remove temporary columns
temp_cols = [col for col in df_features.columns if col.startswith('_')]
df_features = df_features.drop(columns=temp_cols + ['Stiftelsesdato_clean'])

print(f"\nTotal features added: {len(features_added)}")
print(f"Final dataset shape: {df_features.shape}")
print(f"  Rows: {len(df_features):,}")
print(f"  Columns: {len(df_features.columns)}")

# Save dataset
output_parquet = Path('feature_dataset_v1.parquet')
output_csv = Path('feature_dataset_v1.csv')

df_features.to_parquet(output_parquet, index=False)
print(f"\n[OK] Saved: {output_parquet}")
print(f"  Size: {output_parquet.stat().st_size / 1024 / 1024:.2f} MB")

df_features.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"[OK] Saved: {output_csv}")
print(f"  Size: {output_csv.stat().st_size / 1024 / 1024:.2f} MB")

# ============================================================================
# GENERATE DOCUMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING DOCUMENTATION")
print("=" * 80)

# Create calculation log
log_md = []
log_md.append("# Feature Calculation Log - Version 1")
log_md.append(f"Generated: {datetime.now()}")
log_md.append("")
log_md.append("## Source Data")
log_md.append(f"- Input: {input_file}")
log_md.append(f"- Rows: {len(df):,}")
log_md.append(f"- Companies: {df['Orgnr'].nunique():,}")
log_md.append(f"- Years: {sorted(df['year'].unique())}")
log_md.append("")
log_md.append(f"## Features Added: {len(features_added)}")
log_md.append("")

# Group by category
categories = {
    'Financial Ratios': features_added[0:11],
    'Temporal Features': features_added[11:21],
    'Missingness Features': features_added[21:28],
    'Company Characteristics': features_added[28:32],
    'Warning Signals': features_added[32:37]
}

for cat_name, feature_list in categories.items():
    log_md.append(f"### {cat_name} ({len(feature_list)} features)")
    log_md.append("")

    for feature_info in calculation_log:
        if feature_info['name'] in feature_list:
            log_md.append(f"#### {feature_info['name']}")
            log_md.append(f"- **Formula:** `{feature_info['formula']}`")
            log_md.append(f"- **Theory:** {feature_info['theory']}")
            log_md.append(f"- **Calculated for:** {feature_info['calculated']:,} rows")
            log_md.append(f"- **Missing:** {feature_info['missing']:,} rows")
            if feature_info.get('mean') is not None:
                log_md.append(f"- **Mean:** {feature_info['mean']:.4f}")
            if feature_info.get('median') is not None:
                log_md.append(f"- **Median:** {feature_info['median']:.4f}")
            log_md.append("")

log_md.append("## Key Decisions")
log_md.append("")
log_md.append("1. **Division by zero:** Set result to NaN rather than infinity")
log_md.append("2. **Negative values:** Kept as-is (negative equity is informative)")
log_md.append("3. **Temporal features:** Only calculated for companies with consecutive years")
log_md.append("4. **Company age:** Set to NaN if Stiftelsesdato missing")
log_md.append("5. **Log transformations:** Added 1 before log to handle zeros")
log_md.append("")
log_md.append("## Theoretical Foundation")
log_md.append("")
log_md.append("All features based on established bankruptcy prediction literature:")
log_md.append("- **Beaver (1966):** Financial ratios as predictors of failure")
log_md.append("- **Altman (1968):** Z-Score model and discriminant analysis")
log_md.append("- **Ohlson (1980):** Logistic regression bankruptcy model")

# Save calculation log
log_file = Path('feature_calculation_log.md')
with open(log_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_md))

print(f"[OK] Saved: {log_file}")

# Generate statistics report
stats_lines = []
stats_lines.append("=" * 80)
stats_lines.append("FEATURE STATISTICS REPORT")
stats_lines.append("=" * 80)
stats_lines.append(f"Generated: {datetime.now()}")
stats_lines.append("")
stats_lines.append(f"Dataset: feature_dataset_v1.parquet")
stats_lines.append(f"Total rows: {len(df_features):,}")
stats_lines.append(f"Total companies: {df_features['Orgnr'].nunique():,}")
stats_lines.append(f"Years: {sorted(df_features['year'].unique())}")
stats_lines.append("")
stats_lines.append(f"Features added: {len(features_added)}")
stats_lines.append("")

for feature_info in calculation_log:
    stats_lines.append(f"{feature_info['name']}:")
    stats_lines.append(f"  Calculated: {feature_info['calculated']:,}")
    stats_lines.append(f"  Missing: {feature_info['missing']:,}")
    if feature_info.get('mean'):
        stats_lines.append(f"  Mean: {feature_info['mean']:.4f}")
        stats_lines.append(f"  Median: {feature_info['median']:.4f}")
        stats_lines.append(f"  Std: {feature_info['std']:.4f}")
        stats_lines.append(f"  Min: {feature_info['min']:.4f}")
        stats_lines.append(f"  Max: {feature_info['max']:.4f}")
    stats_lines.append("")

stats_file = Path('feature_statistics.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(stats_lines))

print(f"[OK] Saved: {stats_file}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)
print(f"Completed: {datetime.now()}")
print("\nFiles created:")
print(f"  1. {output_parquet}")
print(f"  2. {output_csv}")
print(f"  3. {log_file}")
print(f"  4. {stats_file}")
