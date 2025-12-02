"""
Filing Behavior and Company Size Analysis
==========================================

This script analyzes:
1. How many businesses have NOT delivered their regnskap (financial statements)
2. Which combinations of years are missing (e.g., missing 2016+2018 but not 2017)
3. Company size distribution by employees (small: 1-20, medium: 21-100, large: 100+)
4. Cross-analysis: Filing behavior by company size

Output:
- Detailed statistics on non-filing patterns
- Company size distribution
- Bankruptcy rates by filing behavior and size
- Visualizations and summary tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("="*70)
print("FILING BEHAVIOR AND COMPANY SIZE ANALYSIS")
print("="*70)

# Load data
script_dir = Path(__file__).parent
input_file = script_dir.parent / 'data' / 'processed' / 'norwegian_companies_panel.parquet'

print(f"\nLoading data from: {input_file}")
df = pd.read_parquet(input_file)

print(f"\nDataset shape: {df.shape}")
print(f"Total observations: {len(df):,}")
print(f"Unique companies: {df['Orgnr'].nunique():,}")
print(f"Years covered: {sorted(df['year'].unique())}")

# ============================================================================
# PART 1: FILING BEHAVIOR ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("PART 1: FILING BEHAVIOR ANALYSIS")
print("="*70)

# Create pivot table to see which years each company filed
filing_pivot = df.pivot_table(
    index='Orgnr',
    columns='year',
    values='bankrupt',  # Use any column, we just care if row exists
    aggfunc='count'
).fillna(0).astype(bool)

# Rename columns for clarity
filing_pivot.columns = [f'filed_{year}' for year in filing_pivot.columns]

# Add bankruptcy status (take max since if bankrupt in any year, they're bankrupt)
company_bankruptcy = df.groupby('Orgnr')['bankrupt'].max()
filing_pivot['bankrupt'] = company_bankruptcy

# Calculate filing patterns
filing_pivot['total_years_filed'] = filing_pivot[['filed_2016', 'filed_2017', 'filed_2018']].sum(axis=1)
filing_pivot['filed_all_years'] = (filing_pivot['total_years_filed'] == 3)
filing_pivot['filed_no_years'] = (filing_pivot['total_years_filed'] == 0)

print("\n1. OVERALL FILING STATISTICS")
print("-" * 70)

total_companies = len(filing_pivot)
print(f"Total companies: {total_companies:,}")

filed_all = filing_pivot['filed_all_years'].sum()
filed_none = filing_pivot['filed_no_years'].sum()
filed_partial = total_companies - filed_all - filed_none

print(f"\nFiled all 3 years (2016, 2017, 2018): {filed_all:,} ({filed_all/total_companies*100:.1f}%)")
print(f"Filed no years (never filed): {filed_none:,} ({filed_none/total_companies*100:.1f}%)")
print(f"Filed partially (1-2 years): {filed_partial:,} ({filed_partial/total_companies*100:.1f}%)")

print("\n2. FILING PATTERNS BY NUMBER OF YEARS")
print("-" * 70)

filing_counts = filing_pivot.groupby('total_years_filed').agg({
    'bankrupt': ['count', 'sum', 'mean']
}).round(4)
filing_counts.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
filing_counts['Percentage'] = (filing_counts['Companies'] / total_companies * 100).round(2)

print(filing_counts.to_string())

print("\n3. SPECIFIC FILING COMBINATIONS")
print("-" * 70)

# Create pattern strings
def filing_pattern(row):
    years_filed = []
    if row['filed_2016']:
        years_filed.append('2016')
    if row['filed_2017']:
        years_filed.append('2017')
    if row['filed_2018']:
        years_filed.append('2018')

    if not years_filed:
        return 'Never filed'
    return '+'.join(years_filed)

filing_pivot['pattern'] = filing_pivot.apply(filing_pattern, axis=1)

# Count each pattern
pattern_stats = filing_pivot.groupby('pattern').agg({
    'bankrupt': ['count', 'sum', 'mean']
}).round(4)
pattern_stats.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
pattern_stats = pattern_stats.sort_values('Companies', ascending=False)
pattern_stats['Percentage'] = (pattern_stats['Companies'] / total_companies * 100).round(2)

print("\nAll filing combinations (sorted by frequency):")
print(pattern_stats.to_string())

print("\n4. NON-FILING PATTERNS (excluding those who filed all years)")
print("-" * 70)

non_complete_filers = filing_pivot[~filing_pivot['filed_all_years']].copy()
non_complete_stats = non_complete_filers.groupby('pattern').agg({
    'bankrupt': ['count', 'sum', 'mean']
}).round(4)
non_complete_stats.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
non_complete_stats = non_complete_stats.sort_values('Companies', ascending=False)
non_complete_stats['Percentage'] = (non_complete_stats['Companies'] / len(non_complete_filers) * 100).round(2)

print(f"\nTotal companies with incomplete filing: {len(non_complete_filers):,}")
print(f"Bankruptcy rate (incomplete filers): {non_complete_filers['bankrupt'].mean()*100:.2f}%")
print(f"Bankruptcy rate (complete filers): {filing_pivot[filing_pivot['filed_all_years']]['bankrupt'].mean()*100:.2f}%")
print("\nTop 10 incomplete filing patterns:")
print(non_complete_stats.head(10).to_string())

# ============================================================================
# PART 2: COMPANY SIZE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("PART 2: COMPANY SIZE ANALYSIS (by number of employees)")
print("="*70)

# Get employee data from most recent available year per company
company_employees = df.sort_values('year', ascending=False).groupby('Orgnr').first()['Antall ansatte']

# Convert to numeric (handle any string values)
company_employees = pd.to_numeric(company_employees, errors='coerce')

print("\n1. EMPLOYEE DATA AVAILABILITY")
print("-" * 70)

total_with_employee_data = company_employees.notna().sum()
print(f"Companies with employee data: {total_with_employee_data:,} ({total_with_employee_data/total_companies*100:.1f}%)")
print(f"Companies without employee data: {(total_companies - total_with_employee_data):,} ({(total_companies - total_with_employee_data)/total_companies*100:.1f}%)")

# Categorize by size (only for companies with employee data)
def categorize_size(employees):
    if pd.isna(employees):
        return 'Unknown'
    elif employees == 0:
        return 'Zero employees'
    elif employees <= 20:
        return 'Small (1-20)'
    elif employees <= 100:
        return 'Medium (21-100)'
    else:
        return 'Large (100+)'

filing_pivot['employees'] = company_employees
filing_pivot['size_category'] = filing_pivot['employees'].apply(categorize_size)

print("\n2. COMPANY SIZE DISTRIBUTION")
print("-" * 70)

size_stats = filing_pivot.groupby('size_category').agg({
    'bankrupt': ['count', 'sum', 'mean'],
    'employees': ['mean', 'median', 'min', 'max']
}).round(2)

size_stats.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate',
                      'Avg_Employees', 'Median_Employees', 'Min_Employees', 'Max_Employees']

# Reorder by logical size order
size_order = ['Small (1-20)', 'Medium (21-100)', 'Large (100+)', 'Zero employees', 'Unknown']
size_stats = size_stats.reindex([cat for cat in size_order if cat in size_stats.index])

size_stats['Percentage'] = (size_stats['Companies'] / total_companies * 100).round(2)

print(size_stats.to_string())

# ============================================================================
# PART 3: CROSS-ANALYSIS (Filing Behavior × Company Size)
# ============================================================================

print("\n" + "="*70)
print("PART 3: FILING BEHAVIOR BY COMPANY SIZE")
print("="*70)

print("\n1. FILING COMPLETENESS BY SIZE")
print("-" * 70)

cross_tab = pd.crosstab(
    filing_pivot['size_category'],
    filing_pivot['filed_all_years'],
    margins=True
)
cross_tab.columns = ['Incomplete Filing', 'Complete Filing', 'Total']

# Add percentages
cross_pct = pd.crosstab(
    filing_pivot['size_category'],
    filing_pivot['filed_all_years'],
    normalize='index'
) * 100

print("\nCount:")
print(cross_tab.to_string())
print("\nPercentage (by row):")
print(cross_pct.round(1).to_string())

print("\n2. BANKRUPTCY RATE BY SIZE AND FILING BEHAVIOR")
print("-" * 70)

bankruptcy_cross = filing_pivot.groupby(['size_category', 'filed_all_years'])['bankrupt'].agg(['count', 'sum', 'mean']).round(4)
bankruptcy_cross.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
bankruptcy_cross['Bankruptcy_Rate_Pct'] = (bankruptcy_cross['Bankruptcy_Rate'] * 100).round(2)

print(bankruptcy_cross.to_string())

print("\n3. DETAILED FILING PATTERNS BY SIZE")
print("-" * 70)

# For each size category, show top filing patterns
for size in ['Small (1-20)', 'Medium (21-100)', 'Large (100+)']:
    if size in filing_pivot['size_category'].values:
        print(f"\n{size} companies:")
        size_patterns = filing_pivot[filing_pivot['size_category'] == size].groupby('pattern').agg({
            'bankrupt': ['count', 'sum', 'mean']
        }).round(4)
        size_patterns.columns = ['Companies', 'Bankruptcies', 'Bankruptcy_Rate']
        size_patterns = size_patterns.sort_values('Companies', ascending=False).head(5)
        print(size_patterns.to_string())

# ============================================================================
# PART 4: KEY INSIGHTS
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Insight 1: Filing behavior impact
complete_bankruptcy_rate = filing_pivot[filing_pivot['filed_all_years']]['bankrupt'].mean()
incomplete_bankruptcy_rate = filing_pivot[~filing_pivot['filed_all_years']]['bankrupt'].mean()

print(f"\n1. FILING BEHAVIOR IMPACT:")
print(f"   Companies that filed all years: {complete_bankruptcy_rate*100:.2f}% bankruptcy rate")
print(f"   Companies with incomplete filing: {incomplete_bankruptcy_rate*100:.2f}% bankruptcy rate")
print(f"   Risk multiplier: {incomplete_bankruptcy_rate/complete_bankruptcy_rate:.1f}x higher risk")

# Insight 2: Size impact
small_bankruptcy = filing_pivot[filing_pivot['size_category'] == 'Small (1-20)']['bankrupt'].mean()
medium_bankruptcy = filing_pivot[filing_pivot['size_category'] == 'Medium (21-100)']['bankrupt'].mean()
large_bankruptcy = filing_pivot[filing_pivot['size_category'] == 'Large (100+)']['bankrupt'].mean()

if not pd.isna(small_bankruptcy):
    print(f"\n2. COMPANY SIZE IMPACT:")
    print(f"   Small companies (1-20 employees): {small_bankruptcy*100:.2f}% bankruptcy rate")
    if not pd.isna(medium_bankruptcy):
        print(f"   Medium companies (21-100 employees): {medium_bankruptcy*100:.2f}% bankruptcy rate")
    if not pd.isna(large_bankruptcy):
        print(f"   Large companies (100+ employees): {large_bankruptcy*100:.2f}% bankruptcy rate")

# Insight 3: Combined effect
if 'Small (1-20)' in filing_pivot['size_category'].values:
    small_complete = filing_pivot[(filing_pivot['size_category'] == 'Small (1-20)') &
                                  (filing_pivot['filed_all_years'])]['bankrupt'].mean()
    small_incomplete = filing_pivot[(filing_pivot['size_category'] == 'Small (1-20)') &
                                    (~filing_pivot['filed_all_years'])]['bankrupt'].mean()

    if not pd.isna(small_complete) and not pd.isna(small_incomplete):
        print(f"\n3. COMBINED EFFECT (Small companies):")
        print(f"   Small + Complete filing: {small_complete*100:.2f}% bankruptcy rate")
        print(f"   Small + Incomplete filing: {small_incomplete*100:.2f}% bankruptcy rate")
        print(f"   Risk multiplier: {small_incomplete/small_complete:.1f}x")

# Insight 4: Most at-risk profile
print(f"\n4. HIGHEST RISK PROFILE:")
highest_risk = bankruptcy_cross.nlargest(5, 'Bankruptcy_Rate')
print(highest_risk.to_string())

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = script_dir

# Save filing patterns
filing_pivot.to_csv(output_dir / 'company_filing_patterns.csv', index=True)
print(f"Company filing patterns saved to: company_filing_patterns.csv")

# Save pattern statistics
pattern_stats.to_csv(output_dir / 'filing_pattern_statistics.csv', index=True)
print(f"Filing pattern statistics saved to: filing_pattern_statistics.csv")

# Save size statistics
size_stats.to_csv(output_dir / 'company_size_statistics.csv', index=True)
print(f"Company size statistics saved to: company_size_statistics.csv")

# Save cross-analysis
bankruptcy_cross.to_csv(output_dir / 'size_filing_bankruptcy_analysis.csv', index=True)
print(f"Cross-analysis saved to: size_filing_bankruptcy_analysis.csv")

# Save summary JSON
summary = {
    'analysis_date': datetime.now().isoformat(),
    'total_companies': int(total_companies),
    'filing_behavior': {
        'filed_all_years': {
            'count': int(filed_all),
            'percentage': float(filed_all/total_companies*100),
            'bankruptcy_rate': float(complete_bankruptcy_rate*100)
        },
        'filed_partial': {
            'count': int(filed_partial),
            'percentage': float(filed_partial/total_companies*100),
            'bankruptcy_rate': float(incomplete_bankruptcy_rate*100)
        },
        'filed_never': {
            'count': int(filed_none),
            'percentage': float(filed_none/total_companies*100)
        }
    },
    'company_size': {
        'small_1_20': {
            'count': int(size_stats.loc['Small (1-20)', 'Companies']) if 'Small (1-20)' in size_stats.index else 0,
            'percentage': float(size_stats.loc['Small (1-20)', 'Percentage']) if 'Small (1-20)' in size_stats.index else 0,
            'bankruptcy_rate': float(size_stats.loc['Small (1-20)', 'Bankruptcy_Rate']*100) if 'Small (1-20)' in size_stats.index else 0
        },
        'medium_21_100': {
            'count': int(size_stats.loc['Medium (21-100)', 'Companies']) if 'Medium (21-100)' in size_stats.index else 0,
            'percentage': float(size_stats.loc['Medium (21-100)', 'Percentage']) if 'Medium (21-100)' in size_stats.index else 0,
            'bankruptcy_rate': float(size_stats.loc['Medium (21-100)', 'Bankruptcy_Rate']*100) if 'Medium (21-100)' in size_stats.index else 0
        },
        'large_100_plus': {
            'count': int(size_stats.loc['Large (100+)', 'Companies']) if 'Large (100+)' in size_stats.index else 0,
            'percentage': float(size_stats.loc['Large (100+)', 'Percentage']) if 'Large (100+)' in size_stats.index else 0,
            'bankruptcy_rate': float(size_stats.loc['Large (100+)', 'Bankruptcy_Rate']*100) if 'Large (100+)' in size_stats.index else 0
        }
    },
    'key_insights': {
        'filing_risk_multiplier': float(incomplete_bankruptcy_rate/complete_bankruptcy_rate),
        'companies_with_employee_data': int(total_with_employee_data),
        'employee_data_coverage': float(total_with_employee_data/total_companies*100)
    }
}

with open(output_dir / 'filing_size_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON saved to: filing_size_summary.json")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
