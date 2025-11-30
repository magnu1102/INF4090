import pandas as pd

print("=" * 80)
print("UNDERSTANDING THE ACCOUNTING METRICS")
print("=" * 80)

# Read the konkurser file (it has the clearest structure)
df = pd.read_excel(r'data\konkurser2019.xlsx')

# Extract the accounting metric descriptions
accounting_metrics = {}
for col in df.columns:
    col_str = str(col).strip()
    if 'Tall' in col_str and 'beskrivelse' in col_str:
        # Get the corresponding number column
        num_col = col_str.replace(' beskrivelse', '').strip()
        if num_col in df.columns:
            # Get the description from the first data row
            description = df[col].iloc[1] if len(df) > 1 else "N/A"
            accounting_metrics[num_col] = description

print("\nACCOUNTING METRICS IN THE DATASET:")
print("-" * 80)
print(f"{'Code':<15} | {'Description (Norwegian)'}")
print("-" * 80)

for code, desc in sorted(accounting_metrics.items()):
    desc_str = str(desc).strip()[:60]
    print(f"{code:<15} | {desc_str}")

print("\n\n" + "=" * 80)
print("ENGLISH TRANSLATION OF KEY METRICS:")
print("=" * 80)
translations = {
    'Tall 1340': 'Sales Revenue (Salgsinntekt)',
    'Tall 7709': 'Other Operating Income (Annen driftsinntekt)',
    'Tall 72': 'Total Income (Sum inntekter)',
    'Tall 217': 'Total Fixed Assets (Sum anleggsmidler)',
    'Tall 194': 'Total Current Assets (Sum omløpsmidler)',
    'Tall 86': 'Total Long-term Debt (Sum langsiktig gjeld)',
    'Tall 85': 'Total Short-term Debt (Sum kortsiktig gjeld)',
    'Tall 146': 'Operating Result/EBIT (Driftsresultat)',
    'Tall 17130': 'Total Financial Expenses (Sum finanskostnader)'
}

for code, translation in translations.items():
    print(f"{code:<15} | {translation}")

print("\n\n" + "=" * 80)
print("SAMPLE DATA - First Bankrupt Company:")
print("=" * 80)

# Show a sample company with its accounting data
sample_idx = 1  # Skip header row
# Find column names (they may have trailing spaces)
navn_col = [c for c in df.columns if 'Navn' in str(c)][0]
orgnr_col = [c for c in df.columns if 'Orgnr' in str(c)][0]
naring_col = [c for c in df.columns if 'Beskrivelse til n' in str(c) and 'ringskode' in str(c)][0]

print(f"\nCompany: {str(df.iloc[sample_idx][navn_col]).strip()}")
print(f"Org Nr: {df.iloc[sample_idx][orgnr_col]}")
print(f"Industry: {str(df.iloc[sample_idx][naring_col]).strip()}")
print(f"\nAccounting Numbers:")
for code in translations.keys():
    matching_cols = [c for c in df.columns if code in str(c) and 'beskrivelse' not in str(c)]
    if matching_cols:
        value = df.iloc[sample_idx][matching_cols[0]]
        print(f"  {translations[code]:<45} | {value}")

print("\n\n" + "=" * 80)
print("DATA STRUCTURE SUMMARY:")
print("=" * 80)
print("""
The dataset contains:
1. COMPANY IDENTIFIERS: Organization number (Orgnr), Name, Addresses
2. BUSINESS CLASSIFICATION: Industry codes (Næringskode), Sector codes
3. ADMINISTRATIVE INFO: Registration dates, capital, contact info
4. KEY PERSONNEL: Board leader, accountant, auditor information
5. ACCOUNTING METRICS: Financial statement line items including:
   - Revenue and income items
   - Assets (fixed and current)
   - Liabilities (long-term and short-term debt)
   - Operating results
   - Financial costs

The files represent:
- books2016.xlsx: Non-bankrupt companies' 2016 accounting data
- book2017.xlsx: Non-bankrupt companies' 2017 accounting data
- books2018.xlsx: Non-bankrupt companies' 2018 accounting data
- konkurser2019.xlsx: Companies that went bankrupt in 2019 (historical data)

This is a bankruptcy prediction dataset - comparing financial metrics of
companies that survived vs. those that went bankrupt.
""")
