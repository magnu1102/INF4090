"""
Count Unique Companies in Original Datasets
============================================

This script counts unique organizations (Orgnr) in the original Excel files:
- books2016.xlsx
- books2017.xlsx
- books2018.xlsx
- konkurser2019.xlsx (with tabs for 2016, 2017, 2018)

Output: Unique company counts for each file
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("ORIGINAL DATASET COMPANY COUNTS")
print("="*70)

# Find data files
data_dir = Path(r'C:\Users\magnu\Desktop\AI Management\INF4090\data')

# Non-bankrupt companies files
books2016_path = data_dir / 'books2016.xlsx'
books2017_path = data_dir / 'book2017.xlsx'  # Note: singular "book" not "books"
books2018_path = data_dir / 'books2018.xlsx'

# Bankrupt companies file (with multiple tabs)
konkurser_path = data_dir / 'konkurser2019.xlsx'

print("\n" + "="*70)
print("NON-BANKRUPT COMPANIES (Books files)")
print("="*70)

# Load and count books2016
print("\nLoading books2016.xlsx...")
df_2016 = pd.read_excel(books2016_path, header=1, skiprows=[2])
df_2016.columns = df_2016.columns.str.strip()  # Remove trailing spaces
unique_2016 = df_2016['Orgnr'].nunique()
total_2016 = len(df_2016)
print(f"  Total rows: {total_2016:,}")
print(f"  Unique Orgnr: {unique_2016:,}")

# Load and count books2017
print("\nLoading books2017.xlsx...")
df_2017 = pd.read_excel(books2017_path, header=1, skiprows=[2])
df_2017.columns = df_2017.columns.str.strip()
unique_2017 = df_2017['Orgnr'].nunique()
total_2017 = len(df_2017)
print(f"  Total rows: {total_2017:,}")
print(f"  Unique Orgnr: {unique_2017:,}")

# Load and count books2018
print("\nLoading books2018.xlsx...")
df_2018 = pd.read_excel(books2018_path, header=1, skiprows=[2])
df_2018.columns = df_2018.columns.str.strip()
unique_2018 = df_2018['Orgnr'].nunique()
total_2018 = len(df_2018)
print(f"  Total rows: {total_2018:,}")
print(f"  Unique Orgnr: {unique_2018:,}")

# Combine all non-bankrupt companies
all_non_bankrupt_orgnr = set()
all_non_bankrupt_orgnr.update(df_2016['Orgnr'].dropna().unique())
all_non_bankrupt_orgnr.update(df_2017['Orgnr'].dropna().unique())
all_non_bankrupt_orgnr.update(df_2018['Orgnr'].dropna().unique())

print("\n" + "-"*70)
print("COMBINED NON-BANKRUPT:")
print(f"  Total rows across all years: {total_2016 + total_2017 + total_2018:,}")
print(f"  Unique Orgnr (any year): {len(all_non_bankrupt_orgnr):,}")

print("\n" + "="*70)
print("BANKRUPT COMPANIES (Konkurser2019 file)")
print("="*70)

# Load konkurser2019 - has tabs for each year
print("\nLoading konkurser2019.xlsx...")
print("This file has separate tabs for 2016, 2017, and 2018")

# Load each tab (konkurser has different structure: header=0, skip row 1 which has dashes)
konkurser_2016 = pd.read_excel(konkurser_path, sheet_name='2016', header=0, skiprows=[1])
konkurser_2016.columns = konkurser_2016.columns.str.strip()
konkurser_2017 = pd.read_excel(konkurser_path, sheet_name='2017', header=0, skiprows=[1])
konkurser_2017.columns = konkurser_2017.columns.str.strip()
konkurser_2018 = pd.read_excel(konkurser_path, sheet_name='2018', header=0, skiprows=[1])
konkurser_2018.columns = konkurser_2018.columns.str.strip()

unique_konkurs_2016 = konkurser_2016['Orgnr'].nunique()
unique_konkurs_2017 = konkurser_2017['Orgnr'].nunique()
unique_konkurs_2018 = konkurser_2018['Orgnr'].nunique()

total_konkurs_2016 = len(konkurser_2016)
total_konkurs_2017 = len(konkurser_2017)
total_konkurs_2018 = len(konkurser_2018)

print(f"\n2016 tab:")
print(f"  Total rows: {total_konkurs_2016:,}")
print(f"  Unique Orgnr: {unique_konkurs_2016:,}")

print(f"\n2017 tab:")
print(f"  Total rows: {total_konkurs_2017:,}")
print(f"  Unique Orgnr: {unique_konkurs_2017:,}")

print(f"\n2018 tab:")
print(f"  Total rows: {total_konkurs_2018:,}")
print(f"  Unique Orgnr: {unique_konkurs_2018:,}")

# Combine all bankrupt companies
all_bankrupt_orgnr = set()
all_bankrupt_orgnr.update(konkurser_2016['Orgnr'].dropna().unique())
all_bankrupt_orgnr.update(konkurser_2017['Orgnr'].dropna().unique())
all_bankrupt_orgnr.update(konkurser_2018['Orgnr'].dropna().unique())

print("\n" + "-"*70)
print("COMBINED BANKRUPT:")
print(f"  Total rows across all tabs: {total_konkurs_2016 + total_konkurs_2017 + total_konkurs_2018:,}")
print(f"  Unique Orgnr (any tab): {len(all_bankrupt_orgnr):,}")

print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

# Check overlap
overlap = all_non_bankrupt_orgnr.intersection(all_bankrupt_orgnr)
only_non_bankrupt = all_non_bankrupt_orgnr - all_bankrupt_orgnr
only_bankrupt = all_bankrupt_orgnr - all_non_bankrupt_orgnr

print(f"\nTotal unique companies in NON-BANKRUPT files: {len(all_non_bankrupt_orgnr):,}")
print(f"Total unique companies in BANKRUPT file: {len(all_bankrupt_orgnr):,}")
print(f"\nCompanies in BOTH datasets: {len(overlap):,}")
print(f"Companies ONLY in non-bankrupt files: {len(only_non_bankrupt):,}")
print(f"Companies ONLY in bankrupt file: {len(only_bankrupt):,}")

print(f"\nTotal unique companies across ALL files: {len(all_non_bankrupt_orgnr.union(all_bankrupt_orgnr)):,}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print(f"""
The overlap of {len(overlap):,} companies means:
- These companies appear in BOTH the non-bankrupt files (2016-2018)
  AND the konkurser2019 file
- This makes sense: They filed financial statements in 2016-2018,
  then went bankrupt in 2019
- This is the core of our prediction problem: predicting which
  companies filing in 2016-2018 will bankrupt in 2019

Companies ONLY in bankrupt file ({len(only_bankrupt):,}):
- These companies went bankrupt but had no financial statements
  in the 2016-2018 books files
- They may have filed in earlier years, or never filed
- These are the "missing data" cases we identified

Companies ONLY in non-bankrupt files ({len(only_non_bankrupt):,}):
- These companies filed in 2016-2018 and did NOT go bankrupt in 2019
- These are the "survived" companies
""")

# Save summary
summary_df = pd.DataFrame({
    'Dataset': [
        'books2016.xlsx',
        'books2017.xlsx',
        'books2018.xlsx',
        'konkurser2019.xlsx (2016 tab)',
        'konkurser2019.xlsx (2017 tab)',
        'konkurser2019.xlsx (2018 tab)',
        '---',
        'All non-bankrupt (combined)',
        'All bankrupt (combined)',
        'Companies in both',
        'Companies only non-bankrupt',
        'Companies only bankrupt',
        'Total unique companies'
    ],
    'Total_Rows': [
        total_2016,
        total_2017,
        total_2018,
        total_konkurs_2016,
        total_konkurs_2017,
        total_konkurs_2018,
        None,
        total_2016 + total_2017 + total_2018,
        total_konkurs_2016 + total_konkurs_2017 + total_konkurs_2018,
        None,
        None,
        None,
        None
    ],
    'Unique_Orgnr': [
        unique_2016,
        unique_2017,
        unique_2018,
        unique_konkurs_2016,
        unique_konkurs_2017,
        unique_konkurs_2018,
        None,
        len(all_non_bankrupt_orgnr),
        len(all_bankrupt_orgnr),
        len(overlap),
        len(only_non_bankrupt),
        len(only_bankrupt),
        len(all_non_bankrupt_orgnr.union(all_bankrupt_orgnr))
    ]
})

output_path = Path(__file__).parent / 'original_dataset_counts.csv'
summary_df.to_csv(output_path, index=False)
print(f"\nSummary saved to: {output_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
