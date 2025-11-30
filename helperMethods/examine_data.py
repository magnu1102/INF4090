import pandas as pd
import os

# File paths
files = {
    'books2016': r'data\books2016.xlsx',
    'books2017': r'data\book2017.xlsx',
    'books2018': r'data\books2018.xlsx',
    'konkurser2019': r'data\konkurser2019.xlsx'
}

print("=" * 80)
print("EXAMINING NORWEGIAN BUSINESS ACCOUNTING DATA")
print("=" * 80)

for name, filepath in files.items():
    print(f"\n{'=' * 80}")
    print(f"FILE: {name}")
    print(f"Path: {filepath}")
    print(f"{'=' * 80}")

    # Read the Excel file
    df = pd.read_excel(filepath)

    # Basic information
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nFirst 3 rows:")
    print(df.head(3))

    print(f"\nBasic statistics:")
    print(df.describe())

    print(f"\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

print("\n" + "=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)
