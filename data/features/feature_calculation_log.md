# Feature Calculation Log - Version 1
Generated: 2025-12-03 18:27:25.997584

## Source Data
- Input: ..\processed\norwegian_companies_panel.parquet
- Rows: 280,840
- Companies: 114,848
- Years: [np.int64(2016), np.int64(2017), np.int64(2018)]

## Data Pre-Processing

### Tall 7709 Merge (Annen driftsinntekt)
**Date:** December 3, 2025
**Rationale:** Tall 7709 (Other Operating Income) had 59.2% missing rate, reducing complete cases from 75% to 36%. Field is legitimately missing for companies without secondary revenue streams.

**Implementation:**
```python
# Merge Tall 7709 into Tall 72 (Sum inntekter)
df_features['Tall 72'] = df_features['Tall 72'].fillna(0) + df_features['Tall 7709'].fillna(0)
df_features.loc[df['Tall 72'].isna() & df['Tall 7709'].isna(), 'Tall 72'] = np.nan
```

**Impact:**
- Tall 72 missing rate: Unchanged at 12.8% (Tall 7709 absorbed when present)
- Complete cases (Sector C): 12,408 → 25,783 (+107.8% increase)
- Information preserved: Signal from Tall 7709 retained when present
- Required features: Reduced from 9 to 8 Tall fields

**Note:** All downstream features using Tall 72 now include Other Operating Income when reported.

---

## Features Added: 38

### Financial Ratios (11 features)

#### likviditetsgrad_1
- **Formula:** `Tall 194 / Tall 85`
- **Theory:** Beaver (1966), Altman (1968)
- **Calculated for:** 261,038 rows
- **Missing:** 19,802 rows
- **Mean:** 13.5397

#### total_gjeldsgrad
- **Formula:** `(Tall 86 + Tall 85) / (Tall 217 + Tall 194)`
- **Theory:** Altman (1968), Zmijewski (1984)
- **Calculated for:** 262,500 rows
- **Missing:** 18,340 rows
- **Mean:** 2.8672

#### langsiktig_gjeldsgrad
- **Formula:** `Tall 86 / (Tall 217 + Tall 194)`
- **Theory:** Leverage analysis
- **Calculated for:** 269,349 rows
- **Missing:** 11,491 rows
- **Mean:** 1.3942

#### kortsiktig_gjeldsgrad
- **Formula:** `Tall 85 / (Tall 217 + Tall 194)`
- **Theory:** Short-term liquidity pressure
- **Calculated for:** 262,528 rows
- **Missing:** 18,312 rows
- **Mean:** 1.7095

#### egenkapitalandel
- **Formula:** `1 - total_gjeldsgrad`
- **Theory:** Norwegian accounting standards
- **Calculated for:** 262,500 rows
- **Missing:** 18,340 rows
- **Mean:** -1.8672

#### driftsmargin
- **Formula:** `Tall 146 / Tall 1340`
- **Theory:** Norwegian accounting standards - Driftsresultat / Salgsinntekt
- **Calculated for:** 217,137 rows
- **Missing:** 63,703 rows
- **Mean:** -0.4187

#### driftsrentabilitet
- **Formula:** `Tall 146 / (Tall 217 + Tall 194)`
- **Theory:** Operating ROA - Altman (1968), Beaver (1966)
- **Calculated for:** 268,027 rows
- **Missing:** 12,813 rows
- **Mean:** -0.1986

#### omsetningsgrad
- **Formula:** `Tall 1340 / (Tall 217 + Tall 194)`
- **Theory:** Asset efficiency
- **Calculated for:** 218,666 rows
- **Missing:** 62,174 rows
- **Mean:** 3.2175

#### rentedekningsgrad
- **Formula:** `Tall 146 / Tall 17130`
- **Theory:** Times Interest Earned ratio
- **Calculated for:** 243,400 rows
- **Missing:** 37,440 rows
- **Mean:** 793.6542

#### altman_z_score
- **Formula:** `0.717*X1 + 3.107*X3 + 0.420*X4 + 0.998*X5 (simplified)`
- **Theory:** Altman (1968, revised 1983)
- **Calculated for:** 215,568 rows
- **Missing:** 65,272 rows
- **Mean:** 4.0134

#### omsetningsvekst_1617
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 182,477 rows
- **Missing:** 98,363 rows
- **Mean:** 81.1618

### Temporal Features (10 features)

#### omsetningsvekst_1718
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 182,201 rows
- **Missing:** 98,639 rows
- **Mean:** 3.4447

#### aktivavekst_1617
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 223,479 rows
- **Missing:** 57,361 rows
- **Mean:** 1.3161

#### aktivavekst_1718
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 223,332 rows
- **Missing:** 57,508 rows
- **Mean:** 2.7018

#### gjeldsvekst_1617
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 218,305 rows
- **Missing:** 62,535 rows
- **Mean:** 4.0022

#### gjeldsvekst_1718
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 218,166 rows
- **Missing:** 62,674 rows
- **Mean:** 3.4399

#### fallende_likviditet
- **Formula:** `3-year trend indicator (0/1)`
- **Theory:** Consistent deterioration signal
- **Calculated for:** 276,685 rows
- **Missing:** 4,155 rows
- **Mean:** 0.1524

#### konsistent_underskudd
- **Formula:** `3-year trend indicator (0/1)`
- **Theory:** Consistent deterioration signal
- **Calculated for:** 276,685 rows
- **Missing:** 4,155 rows
- **Mean:** 0.1196

#### økende_gjeldsgrad
- **Formula:** `3-year trend indicator (0/1)`
- **Theory:** Consistent deterioration signal
- **Calculated for:** 276,685 rows
- **Missing:** 4,155 rows
- **Mean:** 0.1583

#### omsetningsvolatilitet
- **Formula:** `std(revenues) / mean(revenues)`
- **Theory:** Business stability indicator
- **Calculated for:** 203,601 rows
- **Missing:** 77,239 rows
- **Mean:** inf

#### levert_alle_år
- **Formula:** `Filing pattern indicator`
- **Theory:** Non-filing as bankruptcy predictor
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.5861

### Missingness Features (7 features)

#### levert_2018
- **Formula:** `Filing pattern indicator`
- **Theory:** Non-filing as bankruptcy predictor
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.7786

#### antall_år_levert
- **Formula:** `Filing pattern indicator`
- **Theory:** Non-filing as bankruptcy predictor
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 2.4453

#### regnskapskomplett
- **Formula:** `All 8 core Tall fields present (0/1)`
- **Theory:** Data quality indicator
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.7492

#### kan_ikke_beregne_likviditet
- **Formula:** `Missing data for ratio calculation (0/1)`
- **Theory:** Missingness indicator
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.0558

#### kan_ikke_beregne_gjeldsgrad
- **Formula:** `Missing data for ratio calculation (0/1)`
- **Theory:** Missingness indicator
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.0559

#### selskapsalder
- **Formula:** `year - year(Stiftelsesdato)`
- **Theory:** Liability of newness (Stinchcombe 1965)
- **Calculated for:** 274,084 rows
- **Missing:** 6,756 rows
- **Mean:** 11.4651

#### nytt_selskap
- **Formula:** `selskapsalder <= 5`
- **Theory:** Young company risk
- **Calculated for:** 274,084 rows
- **Missing:** 6,756 rows
- **Mean:** 0.4249

### Company Characteristics (7 features)

#### log_totalkapital
- **Formula:** `log(Tall 217 + Tall 194 + 1)`
- **Theory:** Company size proxy
- **Calculated for:** 272,271 rows
- **Missing:** 8,569 rows
- **Mean:** -inf

#### log_omsetning
- **Formula:** `log(Tall 1340 + 1)`
- **Theory:** Revenue size
- **Calculated for:** 219,655 rows
- **Missing:** 61,185 rows
- **Mean:** -inf

#### byttet_revisor_1617
- **Formula:** `Auditor changed between years (0/1)`
- **Theory:** Distress signal - auditor changes
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.0597

#### byttet_revisor_1718
- **Formula:** `Auditor changed between years (0/1)`
- **Theory:** Distress signal - auditor changes
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.0562

#### byttet_revisor_noensinne
- **Formula:** `Auditor changed between years (0/1)`
- **Theory:** Distress signal - auditor changes
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.1109

#### negativ_egenkapital
- **Formula:** `egenkapitalandel < 0`
- **Theory:** Technical insolvency
- **Calculated for:** 262,500 rows
- **Missing:** 0 rows

#### sterkt_overbelånt
- **Formula:** `total_gjeldsgrad > 0.8`
- **Theory:** High leverage warning
- **Calculated for:** 262,500 rows
- **Missing:** 0 rows

### Warning Signals (3 features)

#### kan_ikke_dekke_renter
- **Formula:** `rentedekningsgrad < 1.0`
- **Theory:** Interest coverage failure
- **Calculated for:** 243,400 rows
- **Missing:** 0 rows

#### lav_likviditet
- **Formula:** `likviditetsgrad_1 < 1.0`
- **Theory:** Liquidity crisis
- **Calculated for:** 261,038 rows
- **Missing:** 0 rows

#### driftsunderskudd
- **Formula:** `Tall 146 < 0`
- **Theory:** Unprofitable operations
- **Calculated for:** 271,118 rows
- **Missing:** 0 rows

## Key Decisions

1. **Division by zero:** Set result to NaN rather than infinity
2. **Negative values:** Kept as-is (negative equity is informative)
3. **Temporal features:** Only calculated for companies with consecutive years
4. **Company age:** Set to NaN if Stiftelsesdato missing
5. **Log transformations:** Added 1 before log to handle zeros

## Theoretical Foundation

All features based on established bankruptcy prediction literature:
- **Beaver (1966):** Financial ratios as predictors of failure
- **Altman (1968):** Z-Score model and discriminant analysis
- **Ohlson (1980):** Logistic regression bankruptcy model