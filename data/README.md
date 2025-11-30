# Data Documentation

## Directory Structure

```
data/
├── raw/                 # Original downloaded data (not committed to git)
├── processed/           # Cleaned intermediate data (not committed to git)
└── final/               # Analysis-ready datasets (not committed to git)
```

## Raw Data Files

### 1. sp500_esg_data.csv
**Source:** Kaggle - S&P 500 ESG and Stocks Data (2023-24)
**URL:** https://www.kaggle.com/datasets/rikinzala/s-and-p-500-esg-and-stocks-data-2023-24

**Description:** ESG scores for S&P 500 companies as of September 2023

**Columns:**
- `Ticker`: Stock ticker symbol
- `Company`: Company name
- `Sector`: Industry sector
- `ESG_Score`: Overall ESG score (0-100)
- `Environment_Score`: Environmental pillar score
- `Social_Score`: Social pillar score
- `Governance_Score`: Governance pillar score

**Size:** ~400-500 companies

### 2. sp500_price_data.csv
**Source:** Kaggle - S&P 500 ESG and Stocks Data (2023-24)

**Description:** Daily stock prices for S&P 500 companies

**Date Range:** September 2023 - August 2024 (~250 trading days)

**Columns:**
- `Date`: Trading date
- `Ticker`: Stock ticker symbol
- `Open`, `High`, `Low`, `Close`: Daily prices
- `Volume`: Trading volume

**Size:** ~100,000+ records (tickers × trading days)

### 3. DGS3MO.csv
**Source:** Federal Reserve Economic Data (FRED)
**URL:** https://fred.stlouisfed.org/series/DGS3MO

**Description:** U.S. 3-Month Treasury Constant Maturity Rate (used as risk-free rate)

**Date Range:** September 2023 - August 2024

**Columns:**
- `DATE`: Date
- `DGS3MO`: 3-month Treasury rate (annual percentage)

**Size:** ~250 records

### 4. sp500_index.csv
**Source:** Yahoo Finance via yfinance library

**Description:** S&P 500 index daily prices

**Date Range:** September 2023 - August 2024

**Columns:**
- `Date`: Trading date
- `Open`, `High`, `Low`, `Close`, `Adj Close`: Index values
- `Volume`: Trading volume

**Size:** ~250 records

### 5. company_info.csv
**Source:** Yahoo Finance via yfinance library

**Description:** Company metadata (market cap, sector, industry)

**Columns:**
- `Ticker`: Stock ticker symbol
- `Company_Name`: Full company name
- `Sector`: Industry sector
- `Industry`: Specific industry
- `Market_Cap`: Market capitalization (USD)
- `Country`: Country of incorporation

**Size:** ~400-500 companies

## Processed Data Files

### 1. esg_cleaned.csv
**Description:** Cleaned ESG scores with standardized tickers

**Transformations:**
- Removed duplicates
- Standardized ticker symbols (uppercase)
- Handled missing values (<5%: drop, ≥5%: impute with sector median)
- Validated score ranges

### 2. prices_cleaned.csv
**Description:** Cleaned stock price data

**Transformations:**
- Filtered to analysis window (Sept 2023 - Aug 2024)
- Forward-filled missing prices (max 5 days for weekends/holidays)
- Dropped tickers with >10% missing data
- Removed infinite values

### 3. returns.csv
**Description:** Daily stock returns

**Columns:**
- `Date`, `Ticker`, `Close` (price)
- `Return`: Daily simple return = (P_t - P_{t-1}) / P_{t-1}

**Transformations:**
- Calculated daily returns using pct_change()
- Removed first day (NaN from calculation)

### 4. market_returns.csv
**Description:** Daily S&P 500 index returns

**Columns:**
- `Date`
- `Market_Return`: Daily S&P 500 return

### 5. risk_free_rate.csv
**Description:** Daily risk-free rate

**Columns:**
- `Date`
- `Annual_Rate`: Original 3-month T-bill rate (annual %)
- `Daily_RF_Rate`: Converted to daily rate using: (1 + annual/100)^(1/252) - 1

**Transformations:**
- Converted annual rate to daily rate
- Aligned with trading days from stock data
- Forward-filled for missing trading days

### 6. performance_metrics.csv
**Description:** Performance metrics per company (12-month window)

**Columns:**
- `Ticker`
- `Trading_Days`: Number of trading days with data
- `Mean_Daily_Return`: Average daily return
- `Mean_Daily_Excess_Return`: Average (return - risk-free rate)
- `Annualized_Return`: Geometric mean annualized
- `Annualized_Excess_Return`: Mean excess return × 252
- `Sharpe_Ratio`: (Mean excess return / Std excess return) × √252
- `Cumulative_Return`: Total return over period

### 7. risk_metrics.csv
**Description:** Risk metrics per company

**Columns:**
- `Ticker`
- `Volatility`: Annualized std dev of returns (daily_std × √252)
- `Beta`: Systematic risk vs. market (Cov(stock, market) / Var(market))
- `Downside_Deviation`: Annualized semi-deviation (negative returns only)
- `Excess_Return_Std`: Annualized std dev of excess returns
- `VaR_5pct`: 5th percentile of daily returns (Value at Risk)
- `Max_Drawdown`: Maximum peak-to-trough decline

### 8. control_variables.csv
**Description:** Control variables for regression

**Columns:**
- `Ticker`
- `Market_Cap_Billions`: Market cap in billions USD
- `Log_Market_Cap`: Natural log of market cap (in billions)
- `Sector_*`: Dummy variables for each sector (one-hot encoded, first dropped)

## Final Data Files

### 1. master_dataset.csv
**Description:** Combined dataset at ticker-date level

**Rows:** ~100,000-120,000 (tickers × trading days)

**Columns:**
- Date, Ticker, Price, Return
- ESG_Score, Environment_Score, Social_Score, Governance_Score
- Daily_RF_Rate, Excess_Return
- Market_Return
- Market_Cap, Sector, Industry

**Use Case:** Daily-level analysis, time series plots

### 2. analysis_dataset.csv ⭐ MAIN ANALYSIS FILE
**Description:** Firm-level dataset for regression analysis

**Rows:** ~400-500 (one per company)

**Columns:**
- `Ticker`, `Company_Name`
- **ESG Variables:** ESG_Score, Environment_Score, Social_Score, Governance_Score
- **Performance:** Sharpe_Ratio, Annualized_Return, Annualized_Excess_Return, Cumulative_Return
- **Risk:** Volatility, Beta, Downside_Deviation, Max_Drawdown
- **Controls:** Log_Market_Cap, Market_Cap_Billions, Sector_* (dummies)

**Use Case:** Cross-sectional OLS regressions (RQ1, RQ2, RQ3)

## Data Quality Notes

### Missing Values
- **ESG scores:** <5% missing, rows dropped
- **Market cap:** Some companies may have missing values (use caution)
- **Beta:** Requires sufficient market data; some may be NaN

### Outliers
- **Extreme returns:** Returns >100% or <-100% flagged (possible stock splits)
- **Sharpe ratios:** Typically -3 to 3; outliers kept but noted
- **Volatility:** Typically 15-50% annualized; higher values possible

### Data Alignment
- All data aligned to **trading days only** (no weekends/holidays)
- Date range: **Sept 1, 2023 - Aug 31, 2024** (~250 trading days)
- Missing trading days handled via forward-fill (max 5 days)

### Survivorship Bias
- Dataset includes only companies that were in S&P 500 as of Sept 2023
- Companies delisted during analysis period may be excluded
- Minimum 200 trading days required for inclusion in analysis

## Reproducibility

To regenerate all processed and final data files:

```bash
python scripts/download_data.py      # Downloads raw data
python scripts/process_data.py       # Creates processed data
python scripts/run_feature_engineering.py  # Creates final data
```

**Total time:** ~10-15 minutes

## License & Attribution

- **Kaggle data:** GPL 3 License
- **FRED data:** Public Domain
- **Yahoo Finance data:** For academic/research use only

**Citation:**
If using this data in academic work, cite:
- Zala, R. (2024). S&P 500 ESG and Stocks Data (2023-24). Kaggle.
- Federal Reserve Bank of St. Louis. (2024). 3-Month Treasury Constant Maturity Rate [DGS3MO]. FRED.
