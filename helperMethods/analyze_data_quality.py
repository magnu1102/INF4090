import pandas as pd
import numpy as np
from collections import defaultdict

print("=" * 80)
print("DATA QUALITY ANALYSIS - Missing Years & Values")
print("=" * 80)

# Load all data
print("\nLoading data...")
books_2016 = pd.read_excel(r'data\books2016.xlsx', header=1)
books_2017 = pd.read_excel(r'data\book2017.xlsx', header=1)
books_2018 = pd.read_excel(r'data\books2018.xlsx', header=1)

konkurser_2016 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2016', header=0)
konkurser_2017 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2017', header=0)
konkurser_2018 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2018', header=0)

# Find the Orgnr column (may have trailing spaces)
def get_orgnr_col(df):
    return [c for c in df.columns if 'Orgnr' in str(c)][0]

# Get organization numbers for each dataset
print("\n" + "=" * 80)
print("DATASET SIZES")
print("=" * 80)

orgnr_col_books = get_orgnr_col(books_2016)
orgnr_col_konkurser = get_orgnr_col(konkurser_2016)

# Clean org numbers (remove dashes and whitespace)
def clean_orgnr(series):
    return series.astype(str).str.strip().str.replace('-', '').str.replace(' ', '')

books_2016_orgnr = set(clean_orgnr(books_2016[orgnr_col_books]))
books_2017_orgnr = set(clean_orgnr(books_2017[orgnr_col_books]))
books_2018_orgnr = set(clean_orgnr(books_2018[orgnr_col_books]))

konkurser_2016_orgnr = set(clean_orgnr(konkurser_2016[orgnr_col_konkurser]))
konkurser_2017_orgnr = set(clean_orgnr(konkurser_2017[orgnr_col_konkurser]))
konkurser_2018_orgnr = set(clean_orgnr(konkurser_2018[orgnr_col_konkurser]))

print(f"\nNON-BANKRUPT companies:")
print(f"  2016: {len(books_2016_orgnr):,} unique companies")
print(f"  2017: {len(books_2017_orgnr):,} unique companies")
print(f"  2018: {len(books_2018_orgnr):,} unique companies")

print(f"\nBANKRUPT companies (went bankrupt in 2019):")
print(f"  2016 data: {len(konkurser_2016_orgnr):,} unique companies")
print(f"  2017 data: {len(konkurser_2017_orgnr):,} unique companies")
print(f"  2018 data: {len(konkurser_2018_orgnr):,} unique companies")

# Check overlap between years for bankrupt companies
print("\n" + "=" * 80)
print("BANKRUPT COMPANIES - Year Coverage Analysis")
print("=" * 80)

all_bankrupt_orgnr = konkurser_2016_orgnr | konkurser_2017_orgnr | konkurser_2018_orgnr
print(f"\nTotal unique bankrupt companies across all years: {len(all_bankrupt_orgnr):,}")

in_all_three_years = konkurser_2016_orgnr & konkurser_2017_orgnr & konkurser_2018_orgnr
in_two_years = (konkurser_2016_orgnr & konkurser_2017_orgnr) | \
               (konkurser_2017_orgnr & konkurser_2018_orgnr) | \
               (konkurser_2016_orgnr & konkurser_2018_orgnr)
in_one_year_only = all_bankrupt_orgnr - in_two_years

print(f"\nCompanies with data for ALL 3 years (2016, 2017, 2018): {len(in_all_three_years):,}")
print(f"  Percentage: {len(in_all_three_years)/len(all_bankrupt_orgnr)*100:.1f}%")

only_2016_2017 = (konkurser_2016_orgnr & konkurser_2017_orgnr) - konkurser_2018_orgnr
only_2017_2018 = (konkurser_2017_orgnr & konkurser_2018_orgnr) - konkurser_2016_orgnr
only_2016_2018 = (konkurser_2016_orgnr & konkurser_2018_orgnr) - konkurser_2017_orgnr

print(f"\nCompanies with data for exactly 2 years:")
print(f"  2016 & 2017 only: {len(only_2016_2017):,}")
print(f"  2017 & 2018 only: {len(only_2017_2018):,}")
print(f"  2016 & 2018 only: {len(only_2016_2018):,}")

only_2016 = konkurser_2016_orgnr - konkurser_2017_orgnr - konkurser_2018_orgnr
only_2017 = konkurser_2017_orgnr - konkurser_2016_orgnr - konkurser_2018_orgnr
only_2018 = konkurser_2018_orgnr - konkurser_2016_orgnr - konkurser_2017_orgnr

print(f"\nCompanies with data for only 1 year:")
print(f"  Only 2016: {len(only_2016):,}")
print(f"  Only 2017: {len(only_2017):,}")
print(f"  Only 2018: {len(only_2018):,}")

# Check overlap for non-bankrupt companies
print("\n" + "=" * 80)
print("NON-BANKRUPT COMPANIES - Year Coverage Analysis")
print("=" * 80)

all_nonbankrupt_orgnr = books_2016_orgnr | books_2017_orgnr | books_2018_orgnr
print(f"\nTotal unique non-bankrupt companies across all years: {len(all_nonbankrupt_orgnr):,}")

nb_all_three = books_2016_orgnr & books_2017_orgnr & books_2018_orgnr
print(f"\nCompanies with data for ALL 3 years: {len(nb_all_three):,}")
print(f"  Percentage: {len(nb_all_three)/len(all_nonbankrupt_orgnr)*100:.1f}%")

nb_only_2016_2017 = (books_2016_orgnr & books_2017_orgnr) - books_2018_orgnr
nb_only_2017_2018 = (books_2017_orgnr & books_2018_orgnr) - books_2016_orgnr
nb_only_2016_2018 = (books_2016_orgnr & books_2018_orgnr) - books_2017_orgnr

print(f"\nCompanies with data for exactly 2 years:")
print(f"  2016 & 2017 only: {len(nb_only_2016_2017):,}")
print(f"  2017 & 2018 only: {len(nb_only_2017_2018):,}")
print(f"  2016 & 2018 only: {len(nb_only_2016_2018):,}")

nb_only_2016 = books_2016_orgnr - books_2017_orgnr - books_2018_orgnr
nb_only_2017 = books_2017_orgnr - books_2016_orgnr - books_2018_orgnr
nb_only_2018 = books_2018_orgnr - books_2016_orgnr - books_2017_orgnr

print(f"\nCompanies with data for only 1 year:")
print(f"  Only 2016: {len(nb_only_2016):,}")
print(f"  Only 2017: {len(nb_only_2017):,}")
print(f"  Only 2018: {len(nb_only_2018):,}")

# Check for missing accounting values
print("\n" + "=" * 80)
print("MISSING ACCOUNTING VALUES ANALYSIS")
print("=" * 80)

def analyze_missing_values(df, name):
    # Get accounting columns (Tall columns)
    accounting_cols = [c for c in df.columns if 'Tall' in str(c) and 'beskrivelse' not in str(c)]

    print(f"\n{name}:")
    print(f"  Total companies: {len(df)}")
    print(f"  Accounting metrics: {len(accounting_cols)}")

    # Count missing values
    total_cells = len(df) * len(accounting_cols)
    missing_counts = df[accounting_cols].isnull().sum().sum()

    # Also check for empty strings or spaces
    empty_counts = 0
    for col in accounting_cols:
        empty_counts += (df[col].astype(str).str.strip() == '').sum()

    print(f"  Total cells: {total_cells:,}")
    print(f"  Missing (null): {missing_counts:,} ({missing_counts/total_cells*100:.2f}%)")
    print(f"  Empty strings: {empty_counts:,} ({empty_counts/total_cells*100:.2f}%)")

    # Count companies with ANY missing data
    companies_with_missing = df[accounting_cols].isnull().any(axis=1).sum()
    print(f"  Companies with ANY missing accounting data: {companies_with_missing:,} ({companies_with_missing/len(df)*100:.1f}%)")

    # Count completely empty companies
    completely_empty = (df[accounting_cols].isnull().all(axis=1) |
                       (df[accounting_cols].astype(str).apply(lambda x: x.str.strip() == '').all(axis=1))).sum()
    print(f"  Companies with NO accounting data: {completely_empty:,} ({completely_empty/len(df)*100:.1f}%)")

analyze_missing_values(books_2016, "Non-bankrupt 2016")
analyze_missing_values(books_2017, "Non-bankrupt 2017")
analyze_missing_values(books_2018, "Non-bankrupt 2018")
analyze_missing_values(konkurser_2016, "Bankrupt 2016")
analyze_missing_values(konkurser_2017, "Bankrupt 2017")
analyze_missing_values(konkurser_2018, "Bankrupt 2018")

# Summary and recommendations
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print(f"""
COVERAGE STATISTICS:
- Bankrupt companies with 3-year data: {len(in_all_three_years):,} ({len(in_all_three_years)/len(all_bankrupt_orgnr)*100:.1f}%)
- Non-bankrupt companies with 3-year data: {len(nb_all_three):,} ({len(nb_all_three)/len(all_nonbankrupt_orgnr)*100:.1f}%)

POTENTIAL DATASET SIZES:
1. STRICT (only companies with ALL 3 years of data):
   - Bankrupt: {len(in_all_three_years):,}
   - Non-bankrupt: {len(nb_all_three):,}
   - Total: {len(in_all_three_years) + len(nb_all_three):,}
   - Class balance: {len(in_all_three_years)/(len(in_all_three_years) + len(nb_all_three))*100:.2f}% bankrupt

2. MODERATE (companies with at least 2 consecutive years):
   - More companies included, handle missing year as null

3. FLEXIBLE (all companies, missing years = null):
   - Maximum sample size
   - More complex missing data handling needed
""")
