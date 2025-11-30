# ESG Stock Performance Impact Analysis

**Course:** DATA 512
**Author:** Ayush Mall
**Project:** Evaluating the Relationship Between ESG Performance, Stock Returns, and Carbon Footprints

## Overview

This project analyzes the relationship between Environmental, Social, and Governance (ESG) scores and stock market performance for S&P 500 companies. Using data from 2023-24, we investigate whether higher ESG scores correlate with better risk-adjusted returns, lower volatility, and which ESG pillar (E, S, or G) drives these relationships.

### Research Questions

1. **RQ1:** Do companies with higher ESG scores earn higher risk-adjusted returns over the following 12 months?
2. **RQ2:** Do higher ESG scores reduce stock return volatility?
3. **RQ3:** Which ESG pillar (E, S, G) drives risk-adjusted returns and volatility the most?

## Project Structure

```
esg-stock-performace-impact/
├── data/
│   ├── raw/               # Original downloaded data
│   ├── processed/         # Cleaned intermediate data
│   └── final/             # Analysis-ready datasets
├── src/
│   ├── data_acquisition/  # Data download modules
│   ├── data_processing/   # Data cleaning & merging
│   ├── feature_engineering/ # Metrics calculation
│   ├── analysis/          # Statistical models
│   └── visualization/     # Plotting functions
├── scripts/
│   ├── download_data.py   # Master data acquisition script
│   ├── process_data.py    # Data processing pipeline
│   ├── run_analysis.py    # Execute regressions
│   └── generate_report.py # Create visualizations
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_regression_analysis.ipynb
├── outputs/
│   ├── figures/           # All plots
│   └── tables/            # Summary statistics & results
├── .env                   # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ayushmall0710/esg-stock-performace-impact.git
cd esg-stock-performace-impact
```

### 2. Create Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Credentials

#### Kaggle API (Required)
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" to create and copy the API key
3. Copy [.env.template](.env.template) to `.env`:
   ```bash
   cp .env.template .env
   ```
4. Add your API keys to `.env`:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   FRED_API_KEY=your_fred_api_key_here
   ```

#### FRED API (Optional)
1. Sign up for a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Add your API key to `.env`

**Note:** If you don't have a FRED API key, the scripts will guide you to download the data manually.

## Usage

### Option 1: Run Complete Pipeline
```bash
# Download all data
python scripts/download_data.py

# Process data
python scripts/process_data.py

# Run analysis
python scripts/run_analysis.py

# Generate visualizations
python scripts/generate_report.py
```

### Option 2: Use Jupyter Notebooks
```bash
jupyter notebook
# Open notebooks in order: 01 → 02 → 03
```

## Data Sources

1. **S&P 500 ESG and Stocks Data (2023-24)**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/rikinzala/s-and-p-500-esg-and-stocks-data-2023-24)
   - License: GPL 3
   - Contains: ESG scores and daily stock prices for ~400 S&P 500 companies

2. **U.S. 3-Month Treasury Rate (DGS3MO)**
   - Source: [FRED](https://fred.stlouisfed.org/series/DGS3MO)
   - License: Public Domain
   - Used for: Risk-free rate in Sharpe ratio calculations

3. **S&P 500 Index Data**
   - Source: Yahoo Finance via `yfinance`
   - Used for: Market benchmark returns and beta calculations

4. **Market Cap & Sector Data**
   - Source: Yahoo Finance via `yfinance`
   - Used for: Control variables in regressions

## Methodology

We use **Ordinary Least Squares (OLS) regression** to analyze the relationship between ESG scores and financial performance:

- **Dependent Variables:** Sharpe ratio (risk-adjusted returns) and volatility
- **Independent Variables:** ESG score (and E, S, G pillars), log market cap, sector dummies
- **Time Window:** 12 months following ESG snapshot (Sept 2023 - Aug 2024)
- **Diagnostics:** Heteroskedasticity tests, multicollinearity checks (VIF), robustness checks

## Key Metrics

- **Sharpe Ratio:** Risk-adjusted return metric
- **Volatility:** Annualized standard deviation of daily returns
- **Beta:** Systematic risk relative to market
- **Excess Returns:** Returns above risk-free rate

## Results

*(To be updated after analysis is complete)*

## License

This project is for educational purposes as part of DATA 512.

## References

- NYU Stern Center for Sustainable Business & Rockefeller Asset Management (2021). [ESG and Financial Performance](https://www.stern.nyu.edu/sites/default/files/assets/documents/NYU-RAM_ESG-Paper_2021%20Rev_0.pdf)
- CFA Institute. [The Role and Rise of ESG Ratings](https://www.cfainstitute.org/insights/articles/the-role-and-rise-of-esg-ratings)
- Li et al. (2024). ESG and carbon emissions relationship study
