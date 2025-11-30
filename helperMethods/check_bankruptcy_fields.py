import pandas as pd

print("=" * 80)
print("CHECKING BANKRUPTCY-RELATED FIELDS IN MERGED DATASET")
print("=" * 80)

# Load the merged dataset
df = pd.read_parquet(r'data\processed\norwegian_companies_panel.parquet')

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Look for bankruptcy-related columns
print("\n" + "=" * 80)
print("BANKRUPTCY-RELATED COLUMNS")
print("=" * 80)

bankruptcy_keywords = ['konkurs', 'bankrupt', 'oppløst', 'tingrett']

bankruptcy_cols = []
for col in df.columns:
    col_lower = str(col).lower()
    if any(keyword in col_lower for keyword in bankruptcy_keywords):
        bankruptcy_cols.append(col)

print(f"\nFound {len(bankruptcy_cols)} bankruptcy-related columns:")
for col in bankruptcy_cols:
    print(f"\n  Column: {col}")
    print(f"  Data type: {df[col].dtype}")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Value counts:")
    print(df[col].value_counts().head(10))

# Check the 'bankrupt' flag we added
print("\n" + "=" * 80)
print("OUR ADDED 'bankrupt' FLAG")
print("=" * 80)

print(f"\nValue counts:")
print(df['bankrupt'].value_counts())

print(f"\nPercentages:")
print(df['bankrupt'].value_counts(normalize=True) * 100)

# Check if there are companies marked bankrupt=1 that have data in other bankruptcy fields
print("\n" + "=" * 80)
print("CROSS-CHECK: Companies with bankrupt=1")
print("=" * 80)

bankrupt_companies = df[df['bankrupt'] == 1]
print(f"\nTotal rows with bankrupt=1: {len(bankrupt_companies):,}")
print(f"Unique companies with bankrupt=1: {bankrupt_companies['Orgnr'].nunique():,}")

# Check the 'Konkurs' field for bankrupt companies
if 'Konkurs' in df.columns:
    print(f"\n'Konkurs' field values for bankrupt=1 companies:")
    print(bankrupt_companies['Konkurs'].value_counts())

    print(f"\n'Konkurs' field values for bankrupt=0 companies:")
    non_bankrupt = df[df['bankrupt'] == 0]
    print(non_bankrupt['Konkurs'].value_counts())

# Check 'Oppløst' field
if 'Oppløst' in df.columns:
    print(f"\n'Oppløst' field values for bankrupt=1 companies:")
    print(bankrupt_companies['Oppløst'].value_counts())

    print(f"\n'Oppløst' field values for bankrupt=0 companies:")
    print(non_bankrupt['Oppløst'].value_counts())

# Check 'Oversendt tingretten' field
if 'Oversendt tingretten' in df.columns:
    print(f"\n'Oversendt tingretten' field values for bankrupt=1 companies:")
    print(bankrupt_companies['Oversendt tingretten'].value_counts())

    print(f"\n'Oversendt tingretten' field values for bankrupt=0 companies:")
    print(non_bankrupt['Oversendt tingretten'].value_counts())

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
The 'bankrupt' flag we added (0/1) is the PRIMARY indicator:
- bankrupt=1: Company went bankrupt in 2019 (was in konkurser2019.xlsx)
- bankrupt=0: Company did NOT go bankrupt (was in books files)

Other bankruptcy-related fields (Konkurs, Oppløst, Oversendt tingretten)
may exist in the original data but appear to be about PAST bankruptcy status
at the time the data was collected (2016-2018), NOT the 2019 bankruptcy event
we're trying to predict.

For your ML model, use 'bankrupt' as the target variable.
""")
