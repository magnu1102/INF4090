import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("BUILDING CLEAN MERGED DATASET - ORIGINAL NORWEGIAN COLUMN NAMES")
print("=" * 80)

# Create output directory
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Load all data files
# ============================================================================
print("\n[1/6] Loading all Excel files...")

# Non-bankrupt companies
print("  Loading non-bankrupt companies...")
books_2016 = pd.read_excel(r'data\books2016.xlsx', header=1)
books_2017 = pd.read_excel(r'data\book2017.xlsx', header=1)
books_2018 = pd.read_excel(r'data\books2018.xlsx', header=1)

# Bankrupt companies
print("  Loading bankrupt companies...")
konkurser_2016 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2016', header=0)
konkurser_2017 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2017', header=0)
konkurser_2018 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2018', header=0)

print(f"  [OK] Loaded 6 files")
print(f"    Non-bankrupt: {len(books_2016):,} + {len(books_2017):,} + {len(books_2018):,}")
print(f"    Bankrupt: {len(konkurser_2016):,} + {len(konkurser_2017):,} + {len(konkurser_2018):,}")

# ============================================================================
# STEP 2: Standardize column names (clean spaces, keep Norwegian)
# ============================================================================
print("\n[2/6] Cleaning column names (removing trailing spaces)...")

def clean_column_names(df):
    """Remove trailing spaces from column names, keep original Norwegian"""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df

books_2016 = clean_column_names(books_2016)
books_2017 = clean_column_names(books_2017)
books_2018 = clean_column_names(books_2018)
konkurser_2016 = clean_column_names(konkurser_2016)
konkurser_2017 = clean_column_names(konkurser_2017)
konkurser_2018 = clean_column_names(konkurser_2018)

print(f"  [OK] Cleaned column names (removed trailing spaces)")

# ============================================================================
# STEP 3: Add year and bankrupt flag to each dataset
# ============================================================================
print("\n[3/6] Adding year and bankrupt flags...")

books_2016['year'] = 2016
books_2016['bankrupt'] = 0

books_2017['year'] = 2017
books_2017['bankrupt'] = 0

books_2018['year'] = 2018
books_2018['bankrupt'] = 0

konkurser_2016['year'] = 2016
konkurser_2016['bankrupt'] = 1

konkurser_2017['year'] = 2017
konkurser_2017['bankrupt'] = 1

konkurser_2018['year'] = 2018
konkurser_2018['bankrupt'] = 1

print(f"  [OK] Added year and bankrupt flags to all datasets")

# ============================================================================
# STEP 4: Clean organization numbers
# ============================================================================
print("\n[4/6] Cleaning organization numbers...")

def clean_orgnr(series):
    """Clean organization numbers - remove spaces, dashes, keep only digits"""
    return series.astype(str).str.strip().str.replace('-', '').str.replace(' ', '')

# Find Orgnr column (may have slight variations)
def get_orgnr_col(df):
    for col in df.columns:
        if 'orgnr' in col.lower():
            return col
    return None

for df in [books_2016, books_2017, books_2018,
           konkurser_2016, konkurser_2017, konkurser_2018]:
    orgnr_col = get_orgnr_col(df)
    if orgnr_col:
        df[orgnr_col] = clean_orgnr(df[orgnr_col])

print(f"  [OK] Cleaned organization numbers (removed spaces and dashes)")

# ============================================================================
# STEP 5: Merge all datasets
# ============================================================================
print("\n[5/6] Merging all datasets into one panel...")

# Concatenate all datasets
all_data = pd.concat([
    books_2016,
    books_2017,
    books_2018,
    konkurser_2016,
    konkurser_2017,
    konkurser_2018
], ignore_index=True)

print(f"  [OK] Merged all datasets")
print(f"    Total rows: {len(all_data):,}")
print(f"    Total columns: {len(all_data.columns)}")

# Get unique companies
orgnr_col = get_orgnr_col(all_data)
unique_companies = all_data[orgnr_col].nunique()
bankrupt_companies = all_data[all_data['bankrupt'] == 1][orgnr_col].nunique()
non_bankrupt_companies = all_data[all_data['bankrupt'] == 0][orgnr_col].nunique()

print(f"\n    Unique companies: {unique_companies:,}")
print(f"      Bankrupt: {bankrupt_companies:,}")
print(f"      Non-bankrupt: {non_bankrupt_companies:,}")

# Check for companies in both categories (should investigate these)
bankrupt_orgnr = set(all_data[all_data['bankrupt'] == 1][orgnr_col])
non_bankrupt_orgnr = set(all_data[all_data['bankrupt'] == 0][orgnr_col])
overlap = bankrupt_orgnr & non_bankrupt_orgnr

if len(overlap) > 0:
    print(f"\n    [WARNING] {len(overlap)} companies appear in BOTH bankrupt and non-bankrupt files!")
    print(f"      These will be marked as bankrupt (bankrupt=1)")
    # Set all instances of overlapping companies to bankrupt
    all_data.loc[all_data[orgnr_col].isin(overlap), 'bankrupt'] = 1

# ============================================================================
# STEP 6: Clean accounting values (convert empty strings to NaN)
# ============================================================================
print("\n[6/6] Cleaning accounting values...")

# Find accounting columns (columns with "Tall" in the name, excluding beskrivelse)
accounting_columns = [col for col in all_data.columns
                     if 'Tall' in str(col) and 'beskrivelse' not in col.lower()]

print(f"    Found {len(accounting_columns)} accounting columns")

for col in accounting_columns:
    # Replace empty strings and whitespace-only strings with NaN
    all_data[col] = all_data[col].replace(r'^\s*$', np.nan, regex=True)
    # Convert to numeric (will make NaN for non-numeric values)
    all_data[col] = pd.to_numeric(all_data[col], errors='coerce')

print(f"  [OK] Converted accounting columns to numeric")
print(f"    Empty strings -> NaN")

# ============================================================================
# SAVE DATASET
# ============================================================================
print("\n" + "=" * 80)
print("SAVING DATASET")
print("=" * 80)

# Save as CSV first (more forgiving with mixed types)
output_file_csv = output_dir / 'norwegian_companies_panel.csv'
all_data.to_csv(output_file_csv, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility
print(f"[OK] Saved: {output_file_csv}")
print(f"  Format: CSV (UTF-8 with BOM for Excel)")
print(f"  Size: {output_file_csv.stat().st_size / 1024 / 1024:.2f} MB")

# Save as parquet (efficient) - convert object columns to string first
print(f"\nPreparing Parquet file (converting mixed types)...")
all_data_parquet = all_data.copy()

# Convert object columns to string to avoid mixed type issues
for col in all_data_parquet.columns:
    if all_data_parquet[col].dtype == 'object':
        all_data_parquet[col] = all_data_parquet[col].astype(str)

output_file_parquet = output_dir / 'norwegian_companies_panel.parquet'
all_data_parquet.to_parquet(output_file_parquet, index=False)
print(f"[OK] Saved: {output_file_parquet}")
print(f"  Format: Parquet (compressed, preserves types)")
print(f"  Size: {output_file_parquet.stat().st_size / 1024 / 1024:.2f} MB")

# ============================================================================
# GENERATE DATA QUALITY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("DATA QUALITY REPORT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("NORWEGIAN COMPANIES DATASET - DATA QUALITY REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")

report_lines.append("\n" + "=" * 80)
report_lines.append("DATASET SUMMARY")
report_lines.append("=" * 80)
report_lines.append(f"\nTotal rows: {len(all_data):,}")
report_lines.append(f"Total columns: {len(all_data.columns)}")
report_lines.append(f"Years covered: {sorted(all_data['year'].unique())}")
report_lines.append(f"\nUnique companies: {unique_companies:,}")
report_lines.append(f"  Bankrupt (went bankrupt in 2019): {bankrupt_companies:,} ({bankrupt_companies/unique_companies*100:.1f}%)")
report_lines.append(f"  Non-bankrupt (survived): {non_bankrupt_companies:,} ({non_bankrupt_companies/unique_companies*100:.1f}%)")

report_lines.append("\n" + "=" * 80)
report_lines.append("ROWS PER YEAR")
report_lines.append("=" * 80)
for year in sorted(all_data['year'].unique()):
    year_data = all_data[all_data['year'] == year]
    bankrupt_count = len(year_data[year_data['bankrupt'] == 1])
    non_bankrupt_count = len(year_data[year_data['bankrupt'] == 0])
    report_lines.append(f"\n{year}:")
    report_lines.append(f"  Total: {len(year_data):,}")
    report_lines.append(f"  Bankrupt: {bankrupt_count:,}")
    report_lines.append(f"  Non-bankrupt: {non_bankrupt_count:,}")

report_lines.append("\n" + "=" * 80)
report_lines.append("COMPANY COVERAGE BY YEAR")
report_lines.append("=" * 80)

company_year_counts = all_data.groupby(orgnr_col)['year'].nunique()
report_lines.append(f"\nCompanies with data for:")
report_lines.append(f"  All 3 years (2016, 2017, 2018): {(company_year_counts == 3).sum():,}")
report_lines.append(f"  Exactly 2 years: {(company_year_counts == 2).sum():,}")
report_lines.append(f"  Only 1 year: {(company_year_counts == 1).sum():,}")

report_lines.append("\n" + "=" * 80)
report_lines.append("ACCOUNTING DATA COMPLETENESS")
report_lines.append("=" * 80)

for col in accounting_columns:
    total_values = len(all_data)
    missing_values = all_data[col].isna().sum()
    present_values = total_values - missing_values
    report_lines.append(f"\n{col}:")
    report_lines.append(f"  Present: {present_values:,} ({present_values/total_values*100:.1f}%)")
    report_lines.append(f"  Missing: {missing_values:,} ({missing_values/total_values*100:.1f}%)")

report_lines.append("\n" + "=" * 80)
report_lines.append("MISSING DATA BY BANKRUPTCY STATUS")
report_lines.append("=" * 80)

for status, label in [(0, 'Non-bankrupt'), (1, 'Bankrupt')]:
    subset = all_data[all_data['bankrupt'] == status]
    report_lines.append(f"\n{label} companies:")
    for col in accounting_columns:
        missing_pct = subset[col].isna().sum() / len(subset) * 100
        report_lines.append(f"  {col}: {missing_pct:.1f}% missing")

report_lines.append("\n" + "=" * 80)
report_lines.append("COLUMN LIST (Original Norwegian Names)")
report_lines.append("=" * 80)
report_lines.append("\nAll columns in dataset:")
for i, col in enumerate(all_data.columns, 1):
    report_lines.append(f"  {i:2d}. {col}")

# Save report
report_file = output_dir / 'data_quality_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

# Also print to console
for line in report_lines:
    print(line)

print(f"\n[OK] Saved report: {report_file}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nFiles created:")
print(f"  1. {output_file_parquet} (main dataset - Parquet)")
print(f"  2. {output_file_csv} (main dataset - CSV)")
print(f"  3. {report_file} (data quality report)")
print(f"\nNext steps:")
print(f"  - Review the data quality report")
print(f"  - Load the dataset in Python/R for analysis")
print(f"  - Begin exploratory data analysis")
print(f"\nNote: All column names kept in original Norwegian")
