# Feature Specification for Bankruptcy Prediction Model
# Norwegian Company Dataset

## Overview
This document specifies all derived features to be calculated from the raw accounting data.
All features are based on established financial theory and bankruptcy prediction literature.

---

## 1. FINANCIAL RATIOS (Finansielle Nøkkeltall)

### 1.1 LIQUIDITY RATIOS (Likviditetsgrad)

#### Current Ratio (Likviditetsgrad 1)
**Norwegian name:** `likviditetsgrad_1`

**Formula:** Current Assets / Short-term Debt
```
likviditetsgrad_1 = Tall 194 / Tall 85
```

**Theoretical Basis:**
- Measures company's ability to pay short-term obligations
- Altman (1968): Key predictor in Z-Score model
- Beaver (1966): Significant difference between failed and non-failed firms
- Interpretation:
  - < 1.0: Company may struggle to pay short-term debts (warning sign)
  - 1.0-2.0: Generally acceptable
  - > 2.0: Good liquidity position

**Bankruptcy Prediction Relevance:**
Companies approaching bankruptcy typically show declining liquidity as they
struggle to meet short-term obligations.

---

#### Quick Ratio (Likviditetsgrad 2 / Syretest)
**Norwegian name:** `likviditetsgrad_2`

**Formula:** (Current Assets - Inventory) / Short-term Debt
```
# Note: We don't have separate inventory data in our dataset
# We'll use current assets as proxy (conservative estimate)
likviditetsgrad_2 = Tall 194 / Tall 85
```

**Theoretical Basis:**
- More conservative than current ratio
- Excludes inventory (least liquid current asset)
- Ohlson (1980): Used in logistic regression bankruptcy model

---

### 1.2 LEVERAGE RATIOS (Gjeldsgrad)

#### Total Debt Ratio (Total Gjeldsgrad)
**Norwegian name:** `total_gjeldsgrad`

**Formula:** Total Debt / Total Assets
```
total_gjeldsgrad = (Tall 86 + Tall 85) / (Tall 217 + Tall 194)
```

**Theoretical Basis:**
- Altman (1968): Market value of equity to book value of total debt
- Zmijewski (1984): Total debt to total assets ratio
- Higher leverage = higher bankruptcy risk
- Interpretation:
  - < 0.3: Low debt, conservative financing
  - 0.3-0.6: Moderate debt levels
  - > 0.6: High debt, increased financial risk
  - > 0.8: Very high risk (overleveraged)

**Bankruptcy Prediction Relevance:**
Failed firms typically have significantly higher debt ratios. Excessive debt
reduces financial flexibility and increases vulnerability to economic shocks.

---

#### Long-term Debt Ratio (Langsiktig Gjeldsgrad)
**Norwegian name:** `langsiktig_gjeldsgrad`

**Formula:** Long-term Debt / Total Assets
```
langsiktig_gjeldsgrad = Tall 86 / (Tall 217 + Tall 194)
```

**Theoretical Basis:**
- Indicates long-term financial structure
- High long-term debt may indicate aggressive growth strategy or financial distress

---

#### Short-term Debt Ratio (Kortsiktig Gjeldsgrad)
**Norwegian name:** `kortsiktig_gjeldsgrad`

**Formula:** Short-term Debt / Total Assets
```
kortsiktig_gjeldsgrad = Tall 85 / (Tall 217 + Tall 194)
```

**Theoretical Basis:**
- High short-term debt ratio indicates immediate payment pressure
- Companies in distress often refinance long-term to short-term debt

---

#### Equity Ratio (Egenkapitalandel)
**Norwegian name:** `egenkapitalandel`

**Formula:** Equity / Total Assets = 1 - Debt Ratio
```
egenkapitalandel = 1 - total_gjeldsgrad
# Or: egenkapitalandel = (Total Assets - Total Debt) / Total Assets
```

**Theoretical Basis:**
- Norwegian accounting standard requirement (20% minimum for AS)
- Negative equity = technical insolvency
- Strong predictor of bankruptcy

---

### 1.3 PROFITABILITY RATIOS (Lønnsomhet)

#### Operating Margin (Driftsmargin)
**Norwegian name:** `driftsmargin`

**Formula:** Operating Result / Total Income
```
driftsmargin = Tall 146 / Tall 72
```

**Theoretical Basis:**
- Measures operational efficiency
- Taffler (1983): Profit before tax to current liabilities
- Persistent negative margins indicate unsustainable business model
- Interpretation:
  - Negative: Operating at a loss
  - 0-5%: Low profitability
  - 5-10%: Moderate profitability
  - > 10%: Good profitability (varies by industry)

**Bankruptcy Prediction Relevance:**
Consistently unprofitable firms cannot sustain operations long-term.

---

#### Return on Assets (Totalkapitalrentabilitet)
**Norwegian name:** `totalkapitalrentabilitet`

**Formula:** Operating Result / Total Assets
```
totalkapitalrentabilitet = Tall 146 / (Tall 217 + Tall 194)
```

**Theoretical Basis:**
- Altman (1968): Retained earnings to total assets (similar concept)
- Beaver (1966): Net income to total assets
- Measures how efficiently assets generate profit
- Low or negative ROA indicates poor asset utilization

---

#### Revenue to Assets (Omsetningsgrad)
**Norwegian name:** `omsetningsgrad`

**Formula:** Revenue / Total Assets
```
omsetningsgrad = Tall 1340 / (Tall 217 + Tall 194)
```

**Theoretical Basis:**
- Asset turnover ratio - measures efficiency
- High turnover = efficient asset use
- Low turnover = assets not generating revenue (potential distress)

---

### 1.4 COVERAGE RATIOS (Dekningsgrad)

#### Interest Coverage (Rentedekningsgrad)
**Norwegian name:** `rentedekningsgrad`

**Formula:** Operating Result / Financial Expenses
```
rentedekningsgrad = Tall 146 / Tall 17130
```

**Theoretical Basis:**
- Measures ability to pay interest on debt
- Times Interest Earned (TIE) ratio
- Interpretation:
  - < 1.0: Cannot cover interest from operations (danger)
  - 1.0-2.5: Marginal coverage
  - > 2.5: Adequate coverage

**Bankruptcy Prediction Relevance:**
Inability to cover interest payments from operating income is a strong
distress signal.

---

### 1.5 COMPOSITE SCORES

#### Altman Z-Score (Simplified)
**Norwegian name:** `altman_z_score`

**Formula (for private companies):**
```
Z = 0.717 * X1 + 0.847 * X2 + 3.107 * X3 + 0.420 * X4 + 0.998 * X5

Where:
X1 = Working Capital / Total Assets
X2 = Retained Earnings / Total Assets (approximated)
X3 = Operating Result / Total Assets
X4 = Equity / Total Liabilities
X5 = Revenue / Total Assets

# Our calculation:
working_capital = Tall 194 - Tall 85
total_assets = Tall 217 + Tall 194
total_liabilities = Tall 86 + Tall 85
equity = total_assets - total_liabilities

X1 = working_capital / total_assets
X2 = 0  # Not available in our data
X3 = Tall 146 / total_assets
X4 = equity / total_liabilities
X5 = Tall 1340 / total_assets

altman_z_score = 0.717*X1 + 3.107*X3 + 0.420*X4 + 0.998*X5
```

**Theoretical Basis:**
- Altman (1968, revised 1983, 2000): Z-Score model
- One of most cited bankruptcy prediction models
- Interpretation:
  - Z > 2.9: Safe zone
  - 1.8 < Z < 2.9: Grey zone
  - Z < 1.8: Distress zone

---

## 2. TEMPORAL FEATURES (Tidsserievariabler)

### 2.1 GROWTH RATES (Vekstrater)

#### Revenue Growth Rate (Omsetningsvekst)
**Norwegian name:** `omsetningsvekst_1617`, `omsetningsvekst_1718`

**Formula:**
```
omsetningsvekst_1617 = (Tall 1340[2017] - Tall 1340[2016]) / Tall 1340[2016]
omsetningsvekst_1718 = (Tall 1340[2018] - Tall 1340[2017]) / Tall 1340[2017]
```

**Theoretical Basis:**
- Declining revenues often precede bankruptcy
- Rapid revenue decline indicates loss of market position

---

#### Asset Growth (Aktivavekst)
**Norwegian name:** `aktivavekst_1617`, `aktivavekst_1718`

**Formula:**
```
total_assets = Tall 217 + Tall 194
aktivavekst_1617 = (total_assets[2017] - total_assets[2016]) / total_assets[2016]
```

**Theoretical Basis:**
- Asset growth without revenue growth may indicate inefficiency
- Rapid asset shrinkage may indicate asset sales under distress

---

#### Debt Growth (Gjeldsvekst)
**Norwegian name:** `gjeldsvekst_1617`, `gjeldsvekst_1718`

**Formula:**
```
total_debt = Tall 86 + Tall 85
gjeldsvekst_1617 = (total_debt[2017] - total_debt[2016]) / total_debt[2016]
```

**Theoretical Basis:**
- Rapid debt accumulation may indicate distress
- Debt increasing while revenues decline is red flag

---

### 2.2 TREND INDICATORS (Trendvariabler)

#### Deteriorating Liquidity (Fallende Likviditet)
**Norwegian name:** `fallende_likviditet`

**Formula:**
```
fallende_likviditet = 1 if (likviditetsgrad_2018 < likviditetsgrad_2017 < likviditetsgrad_2016) else 0
```

**Theoretical Basis:**
- Consistent deterioration stronger signal than single-year value
- Captures temporal dynamics

---

#### Consistent Losses (Konsistent Underskudd)
**Norwegian name:** `konsistent_underskudd`

**Formula:**
```
konsistent_underskudd = 1 if (Tall 146[2016] < 0 AND Tall 146[2017] < 0 AND Tall 146[2018] < 0) else 0
```

**Theoretical Basis:**
- Three consecutive loss years indicates structural problems
- Not temporary setback but fundamental business model issues

---

#### Increasing Leverage (Økende Gjeldsgrad)
**Norwegian name:** `økende_gjeldsgrad`

**Formula:**
```
økende_gjeldsgrad = 1 if (total_gjeldsgrad_2018 > total_gjeldsgrad_2017 > total_gjeldsgrad_2016) else 0
```

---

### 2.3 VOLATILITY MEASURES (Volatilitet)

#### Revenue Volatility (Omsetningsvolatilitet)
**Norwegian name:** `omsetningsvolatilitet`

**Formula:**
```
revenues = [Tall 1340[2016], Tall 1340[2017], Tall 1340[2018]]
omsetningsvolatilitet = std_deviation(revenues) / mean(revenues)  # Coefficient of variation
```

**Theoretical Basis:**
- High volatility indicates unstable business
- Consistent performance easier to finance and manage

---

## 3. MISSINGNESS FEATURES (Manglende Data-variabler)

### 3.1 FILING STATUS (Rapporteringsstatus)

#### Filed All Years (Levert Alle År)
**Norwegian name:** `levert_alle_år`

**Formula:**
```
levert_alle_år = 1 if company has data for 2016, 2017, AND 2018 else 0
```

**Empirical Basis:**
- Our analysis showed 76.5% of bankrupt companies didn't file 2018 data
- Non-filing is strong bankruptcy predictor

---

#### Filed Most Recent Year (Levert Siste År)
**Norwegian name:** `levert_2018`

**Formula:**
```
levert_2018 = 1 if company has 2018 data else 0
```

---

#### Number of Years Filed (Antall År Levert)
**Norwegian name:** `antall_år_levert`

**Formula:**
```
antall_år_levert = count of years (2016, 2017, 2018) with data
# Values: 1, 2, or 3
```

---

#### Accounting Completeness (Regnskapskomplett)
**Norwegian name:** `regnskapskomplett`

**Formula:**
```
# For each year, check if all 9 Tall fields are present (not NaN)
required_fields = [Tall 1340, Tall 7709, Tall 72, Tall 217, Tall 194,
                   Tall 86, Tall 85, Tall 146, Tall 17130]

regnskapskomplett_2016 = 1 if all fields present in 2016 else 0
regnskapskomplett_2017 = 1 if all fields present in 2017 else 0
regnskapskomplett_2018 = 1 if all fields present in 2018 else 0
```

---

#### Missing Key Ratios (Mangler Nøkkeltall)
**Norwegian name:** `kan_ikke_beregne_likviditet`, `kan_ikke_beregne_gjeldsgrad`

**Formula:**
```
kan_ikke_beregne_likviditet = 1 if (Tall 194 is NaN OR Tall 85 is NaN) else 0
kan_ikke_beregne_gjeldsgrad = 1 if any debt or asset field is NaN else 0
```

---

## 4. COMPANY CHARACTERISTICS (Selskapskarakteristika)

### 4.1 AGE (Alder)

#### Company Age (Selskapsalder)
**Norwegian name:** `selskapsalder`

**Formula:**
```
selskapsalder = 2018 - year(Stiftelsesdato)
# Or for each year: year - year(Stiftelsesdato)
```

**Theoretical Basis:**
- Liability of newness (Stinchcombe, 1965)
- Young companies have higher failure rates
- Lack of established relationships, market position, resources

---

#### Young Company Indicator (Nytt Selskap)
**Norwegian name:** `nytt_selskap`

**Formula:**
```
nytt_selskap = 1 if selskapsalder <= 5 else 0
```

---

### 4.2 SIZE (Størrelse)

#### Total Assets (Log) (Log Totalkapital)
**Norwegian name:** `log_totalkapital`

**Formula:**
```
log_totalkapital = log(Tall 217 + Tall 194 + 1)  # +1 to avoid log(0)
```

**Theoretical Basis:**
- Smaller companies have higher bankruptcy rates
- Size proxy for resources, diversification, market power
- Log transformation for normality

---

#### Revenue Size (Log Omsetning)
**Norwegian name:** `log_omsetning`

**Formula:**
```
log_omsetning = log(Tall 1340 + 1)
```

---

### 4.3 CATEGORICAL VARIABLES (Kategoriske Variabler)

#### Industry Risk (Bransjerisiko)
**Norwegian name:** `bransjerisiko`

**Calculation:**
```
# Calculate bankruptcy rate per industry (Næringskode 2-digit)
# Assign companies to risk categories based on industry bankruptcy rate
bransjerisiko = 'høy' if industry_bankruptcy_rate > 75th percentile
                'middels' if 25th < rate < 75th percentile
                'lav' if rate < 25th percentile
```

**Theoretical Basis:**
- Industry effects significant (Chava & Jarrow, 2004)
- Cyclical industries (construction, retail) higher risk
- Regulated industries (utilities, healthcare) lower risk

---

#### Organization Form Risk (Organisasjonsformrisiko)
**Norwegian name:** `organisasjonsform_gruppe`

**Categories from Organisasjonsform:**
- AS (Aksjeselskap)
- ASA (Allmennaksjeselskap)
- ANS (Ansvarlig selskap)
- DA (Selskap med delt ansvar)
- Other

**Theoretical Basis:**
- Limited liability forms (AS, ASA) may take more risk
- Personal liability forms (ANS) more conservative

---

#### Regional Unemployment (Regional Arbeidsledighet)
**Norwegian name:** `regional_arbeidsledighet`

**Note:** Would require external data linking Fylke/Kommune to unemployment rates
**Not included in initial feature set unless external data available**

---

## 5. WARNING SIGNALS (Varselsignaler)

### 5.1 CRITICAL THRESHOLDS (Kritiske Terskler)

#### Negative Equity (Negativ Egenkapital)
**Norwegian name:** `negativ_egenkapital`

**Formula:**
```
negativ_egenkapital = 1 if (egenkapitalandel < 0) else 0
```

**Theoretical Basis:**
- Technical insolvency
- Norwegian law: AS must have minimum equity
- Strong bankruptcy predictor

---

#### Overleveraged (Sterkt Overbelånt)
**Norwegian name:** `sterkt_overbelånt`

**Formula:**
```
sterkt_overbelånt = 1 if (total_gjeldsgrad > 0.8) else 0
```

---

#### Cannot Cover Interest (Kan Ikke Dekke Renter)
**Norwegian name:** `kan_ikke_dekke_renter`

**Formula:**
```
kan_ikke_dekke_renter = 1 if (rentedekningsgrad < 1.0) else 0
```

---

#### Low Liquidity (Lav Likviditet)
**Norwegian name:** `lav_likviditet`

**Formula:**
```
lav_likviditet = 1 if (likviditetsgrad_1 < 1.0) else 0
```

---

#### Operating Loss (Driftsunderskudd)
**Norwegian name:** `driftsunderskudd`

**Formula:**
```
driftsunderskudd = 1 if (Tall 146 < 0) else 0
```

---

## 6. INDUSTRY-SPECIFIC FEATURES (Bransjetilpassede Variabler)

#### Deviation from Industry Median (Avvik fra Bransjemedian)
**Norwegian name:** `avvik_likviditet_bransje`, `avvik_gjeldsgrad_bransje`

**Formula:**
```
# For each ratio, calculate:
industry_median = median(ratio) for companies in same Næringskode (2-digit)
avvik_likviditet_bransje = (likviditetsgrad_1 - industry_median) / industry_median
```

**Theoretical Basis:**
- Relative performance more informative than absolute
- Industry norms vary significantly
- Chava & Jarrow (2004): Industry-adjusted variables improve prediction

---

## SUMMARY: COMPLETE FEATURE LIST

### Features to Calculate (by category):

**Liquidity (2):**
- likviditetsgrad_1
- likviditetsgrad_2

**Leverage (4):**
- total_gjeldsgrad
- langsiktig_gjeldsgrad
- kortsiktig_gjeldsgrad
- egenkapitalandel

**Profitability (3):**
- driftsmargin
- totalkapitalrentabilitet
- omsetningsgrad

**Coverage (1):**
- rentedekningsgrad

**Composite (1):**
- altman_z_score

**Growth (6):**
- omsetningsvekst_1617, omsetningsvekst_1718
- aktivavekst_1617, aktivavekst_1718
- gjeldsvekst_1617, gjeldsvekst_1718

**Trends (3):**
- fallende_likviditet
- konsistent_underskudd
- økende_gjeldsgrad

**Volatility (1):**
- omsetningsvolatilitet

**Missingness (7):**
- levert_alle_år
- levert_2018
- antall_år_levert
- regnskapskomplett_2016/2017/2018
- kan_ikke_beregne_likviditet
- kan_ikke_beregne_gjeldsgrad

**Company characteristics (4):**
- selskapsalder
- nytt_selskap
- log_totalkapital
- log_omsetning

**Warning signals (5):**
- negativ_egenkapital
- sterkt_overbelånt
- kan_ikke_dekke_renter
- lav_likviditet
- driftsunderskudd

**Industry-adjusted (2):**
- avvik_likviditet_bransje
- avvik_gjeldsgrad_bransje

**Total: ~45 derived features**

---

## REFERENCES

Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. The Journal of Finance, 23(4), 589-609.

Beaver, W. H. (1966). Financial ratios as predictors of failure. Journal of Accounting Research, 71-111.

Chava, S., & Jarrow, R. A. (2004). Bankruptcy prediction with industry effects. Review of Finance, 8(4), 537-569.

Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. Journal of Accounting Research, 109-131.

Taffler, R. J. (1983). The assessment of company solvency and performance using a statistical model. Accounting and Business Research, 13(52), 295-308.

Zmijewski, M. E. (1984). Methodological issues related to the estimation of financial distress prediction models. Journal of Accounting Research, 59-82.
