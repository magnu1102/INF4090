import pandas as pd

print("=" * 80)
print("DETAILED STRUCTURE ANALYSIS")
print("=" * 80)

# First, examine the konkurser2019 file which has proper column names
print("\n\nANALYZING KONKURSER2019 (Bankruptcies) - This file has proper headers")
print("=" * 80)
konkurser = pd.read_excel(r'data\konkurser2019.xlsx')

print(f"\nShape: {konkurser.shape}")
print(f"Number of bankrupt businesses: {konkurser.shape[0]}")

# Show columns more clearly
print("\n\nCOLUMN STRUCTURE:")
print("-" * 80)
for i, col in enumerate(konkurser.columns, 1):
    col_clean = col.strip()
    sample_value = konkurser[col].iloc[2] if len(konkurser) > 2 else "N/A"
    print(f"{i:2d}. {col_clean[:60]:<60} | Example: {str(sample_value)[:40]}")

# Now read the non-bankrupt files with proper header handling
print("\n\n" + "=" * 80)
print("ANALYZING NON-BANKRUPT FILES (books2016, books2017, books2018)")
print("=" * 80)

# Read with header row
books2016 = pd.read_excel(r'data\books2016.xlsx')
print(f"\nBOOKS 2016 - First few actual column headers:")
print(books2016.iloc[0].tolist()[:10])

print(f"\nBOOKS 2016 Statistics:")
print(f"  Total rows (including header rows): {books2016.shape[0]}")
print(f"  Actual data rows: {books2016.shape[0] - 2}")
print(f"  Number of columns: {books2016.shape[1]}")

books2017 = pd.read_excel(r'data\book2017.xlsx')
print(f"\nBOOKS 2017 Statistics:")
print(f"  Total rows (including header rows): {books2017.shape[0]}")
print(f"  Actual data rows: {books2017.shape[0] - 2}")

books2018 = pd.read_excel(r'data\books2018.xlsx')
print(f"\nBOOKS 2018 Statistics:")
print(f"  Total rows (including header rows): {books2018.shape[0]}")
print(f"  Actual data rows: {books2018.shape[0] - 2}")

# Show key accounting metrics in konkurser2019
print("\n\n" + "=" * 80)
print("KEY ACCOUNTING METRICS (Column names from konkurser2019):")
print("=" * 80)

accounting_cols = [col for col in konkurser.columns if 'Tall' in str(col) or 'beskrivelse' in str(col)]
for col in accounting_cols:
    col_clean = col.strip()
    print(f"  - {col_clean}")

print("\n\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Non-bankrupt businesses 2016: {books2016.shape[0] - 2:,}")
print(f"Non-bankrupt businesses 2017: {books2017.shape[0] - 2:,}")
print(f"Non-bankrupt businesses 2018: {books2018.shape[0] - 2:,}")
print(f"Bankrupt businesses 2019: {konkurser.shape[0] - 1:,}")
print(f"\nTotal columns per file: {konkurser.shape[1]}")
print("\nThese files contain Norwegian business data with:")
print("  - Company identifiers (Orgnr)")
print("  - Company information (Name, Address, Industry codes)")
print("  - Accounting numbers (Tall = Number)")
print("  - Multiple years of data for time-series analysis")
