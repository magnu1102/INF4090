# Feature Calculation Log - Version 1
Generated: 2025-12-03 12:15:02.542962

## Source Data
- Input: ..\processed\norwegian_companies_panel.parquet
- Rows: 280,840
- Companies: 114,848
- Years: [np.int64(2016), np.int64(2017), np.int64(2018)]

## Features Added: 38

### Financial Ratios (11 features)

#### likviditetsgrad_1
- **Formula:** `Tall 194 / Tall 85`
- **Theory:** Beaver (1966), Altman (1968)
- **Calculated for:** 263,917 rows
- **Missing:** 16,923 rows
- **Mean:** 226.0420

#### total_gjeldsgrad
- **Formula:** `(Tall 86 + Tall 85) / (Tall 217 + Tall 194)`
- **Theory:** Altman (1968), Zmijewski (1984)
- **Calculated for:** 264,202 rows
- **Missing:** 16,638 rows
- **Mean:** 750.6455

#### langsiktig_gjeldsgrad
- **Formula:** `Tall 86 / (Tall 217 + Tall 194)`
- **Theory:** Leverage analysis
- **Calculated for:** 271,285 rows
- **Missing:** 9,555 rows
- **Mean:** 509.2214

#### kortsiktig_gjeldsgrad
- **Formula:** `Tall 85 / (Tall 217 + Tall 194)`
- **Theory:** Short-term liquidity pressure
- **Calculated for:** 264,230 rows
- **Missing:** 16,610 rows
- **Mean:** 256.9473

#### egenkapitalandel
- **Formula:** `1 - total_gjeldsgrad`
- **Theory:** Norwegian accounting standards
- **Calculated for:** 264,202 rows
- **Missing:** 16,638 rows
- **Mean:** -749.6455

#### driftsmargin
- **Formula:** `Tall 146 / Tall 1340`
- **Theory:** Norwegian accounting standards - Driftsresultat / Salgsinntekt
- **Calculated for:** 217,567 rows
- **Missing:** 63,273 rows
- **Mean:** 7.3217

#### driftsrentabilitet
- **Formula:** `Tall 146 / (Tall 217 + Tall 194)`
- **Theory:** Operating ROA - Altman (1968), Beaver (1966)
- **Calculated for:** 269,865 rows
- **Missing:** 10,975 rows
- **Mean:** -16.7187

#### omsetningsgrad
- **Formula:** `Tall 1340 / (Tall 217 + Tall 194)`
- **Theory:** Asset efficiency
- **Calculated for:** 219,422 rows
- **Missing:** 61,418 rows
- **Mean:** 13.1777

#### rentedekningsgrad
- **Formula:** `Tall 146 / Tall 17130`
- **Theory:** Times Interest Earned ratio
- **Calculated for:** 245,622 rows
- **Missing:** 35,218 rows
- **Mean:** 771.0529

#### altman_z_score
- **Formula:** `0.717*X1 + 3.107*X3 + 0.420*X4 + 0.998*X5 (simplified)`
- **Theory:** Altman (1968, revised 1983)
- **Calculated for:** 217,220 rows
- **Missing:** 63,620 rows
- **Mean:** -22.6033

#### omsetningsvekst_1617
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 182,599 rows
- **Missing:** 98,241 rows
- **Mean:** 80.9658

### Temporal Features (10 features)

#### omsetningsvekst_1718
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 182,329 rows
- **Missing:** 98,511 rows
- **Mean:** 3.1732

#### aktivavekst_1617
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 224,369 rows
- **Missing:** 56,471 rows
- **Mean:** 32.1228

#### aktivavekst_1718
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 224,190 rows
- **Missing:** 56,650 rows
- **Mean:** 658.6408

#### gjeldsvekst_1617
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 219,152 rows
- **Missing:** 61,688 rows
- **Mean:** 6.3695

#### gjeldsvekst_1718
- **Formula:** `Year-over-year change`
- **Theory:** Temporal dynamics
- **Calculated for:** 218,982 rows
- **Missing:** 61,858 rows
- **Mean:** 56.8854

#### fallende_likviditet
- **Formula:** `3-year trend indicator (0/1)`
- **Theory:** Consistent deterioration signal
- **Calculated for:** 276,685 rows
- **Missing:** 4,155 rows
- **Mean:** 0.1547

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
- **Mean:** 0.1613

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
- **Formula:** `All 9 Tall fields present (0/1)`
- **Theory:** Data quality indicator
- **Calculated for:** 280,840 rows
- **Missing:** 0 rows
- **Mean:** 0.3481

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
- **Calculated for:** 264,202 rows
- **Missing:** 0 rows

#### sterkt_overbelånt
- **Formula:** `total_gjeldsgrad > 0.8`
- **Theory:** High leverage warning
- **Calculated for:** 264,202 rows
- **Missing:** 0 rows

### Warning Signals (3 features)

#### kan_ikke_dekke_renter
- **Formula:** `rentedekningsgrad < 1.0`
- **Theory:** Interest coverage failure
- **Calculated for:** 245,622 rows
- **Missing:** 0 rows

#### lav_likviditet
- **Formula:** `likviditetsgrad_1 < 1.0`
- **Theory:** Liquidity crisis
- **Calculated for:** 263,917 rows
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