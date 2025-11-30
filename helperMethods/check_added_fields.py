import pandas as pd

print("=" * 80)
print("CHECKING WHICH FIELDS WERE ADDED VS. ORIGINAL DATA")
print("=" * 80)

# Load original files to see their columns
print("\n[1] Loading original files...")
books_2016 = pd.read_excel(r'data\books2016.xlsx', header=1)
konkurser_2016 = pd.read_excel(r'data\konkurser2019.xlsx', sheet_name='2016', header=0)

# Clean column names (strip spaces)
books_2016.columns = books_2016.columns.str.strip()
konkurser_2016.columns = konkurser_2016.columns.str.strip()

print(f"Books 2016 columns: {len(books_2016.columns)}")
print(f"Konkurser 2016 columns: {len(konkurser_2016.columns)}")

# Load merged dataset
merged = pd.read_parquet(r'data\processed\norwegian_companies_panel.parquet')
print(f"\nMerged dataset columns: {len(merged.columns)}")

# Find columns in merged that are NOT in original files
print("\n" + "=" * 80)
print("COLUMNS ADDED (NOT IN ORIGINAL DATA)")
print("=" * 80)

original_cols_books = set(books_2016.columns)
original_cols_konkurser = set(konkurser_2016.columns)
all_original_cols = original_cols_books | original_cols_konkurser

merged_cols = set(merged.columns)

added_cols = merged_cols - all_original_cols

print(f"\nColumns we ADDED to the dataset:")
for i, col in enumerate(sorted(added_cols), 1):
    print(f"  {i}. {col}")
    # Show some info about this column
    print(f"     Type: {merged[col].dtype}")
    print(f"     Unique values: {merged[col].nunique()}")
    if merged[col].nunique() < 20:
        print(f"     Values: {sorted(merged[col].unique())}")

# Double check: are there any columns in original that are NOT in merged?
print("\n" + "=" * 80)
print("COLUMNS FROM ORIGINAL DATA THAT ARE MISSING IN MERGED")
print("=" * 80)

missing_from_books = original_cols_books - merged_cols
missing_from_konkurser = original_cols_konkurser - merged_cols

if missing_from_books:
    print(f"\nColumns from books files that are missing:")
    for col in sorted(missing_from_books):
        print(f"  - {col}")
else:
    print(f"\nAll columns from books files are present in merged dataset")

if missing_from_konkurser:
    print(f"\nColumns from konkurser files that are missing:")
    for col in sorted(missing_from_konkurser):
        print(f"  - {col}")
else:
    print(f"\nAll columns from konkurser files are present in merged dataset")

# Check if Orgnr was modified
print("\n" + "=" * 80)
print("MODIFICATIONS TO EXISTING FIELDS")
print("=" * 80)

print(f"\nOrgnr field:")
print(f"  Original (books_2016) - sample values: {books_2016['Orgnr'].head(3).tolist()}")
print(f"  Merged - sample values: {merged['Orgnr'].head(3).tolist()}")
print(f"  Modification: Cleaned (removed spaces and dashes)")

# Check accounting fields
accounting_cols = [col for col in merged.columns if 'Tall' in str(col) and 'beskrivelse' not in col.lower()]
print(f"\nAccounting fields (Tall columns):")
print(f"  Number of fields: {len(accounting_cols)}")
print(f"  Modification: Empty strings converted to NaN, converted to numeric type")
print(f"  Example - Tall 1340:")
print(f"    Original type: {books_2016['Tall 1340'].dtype}")
print(f"    Merged type: {merged['Tall 1340'].dtype}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
FIELDS WE ADDED:
{chr(10).join('  - ' + col for col in sorted(added_cols))}

MODIFICATIONS TO ORIGINAL FIELDS:
  - Orgnr: Cleaned (removed spaces, dashes)
  - Accounting columns (Tall *): Empty strings -> NaN, converted to numeric
  - Column names: Stripped trailing spaces

ALL OTHER FIELDS ARE ORIGINAL DATA FROM THE SOURCE FILES.
""")
