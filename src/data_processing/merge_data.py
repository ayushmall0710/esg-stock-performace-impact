"""
Merge all datasets into master analysis file.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def merge_all_data(esg_file: str = "data/processed/esg_cleaned.csv",
                   returns_file: str = "data/processed/returns.csv",
                   market_returns_file: str = "data/processed/market_returns.csv",
                   risk_free_file: str = "data/processed/risk_free_rate.csv",
                   company_info_file: str = "data/raw/company_info.csv",
                   output_file: str = "data/final/master_dataset.csv") -> pd.DataFrame:
    """
    Merge all datasets into a single master file.

    Merge steps:
    1. Start with returns data (ticker-date level)
    2. Merge with ESG scores (ticker level)
    3. Merge with risk-free rate (date level)
    4. Merge with market returns (date level)
    5. Merge with company info (ticker level)
    6. Final validation

    Args:
        esg_file: Path to cleaned ESG data
        returns_file: Path to stock returns data
        market_returns_file: Path to market returns data
        risk_free_file: Path to risk-free rate data
        company_info_file: Path to company information
        output_file: Path to save master dataset

    Returns:
        Merged DataFrame
    """
    print("=" * 60)
    print("Merging All Datasets")
    print("=" * 60)

    # Step 1: Load returns data (base dataset)
    print("\n📂 Step 1: Loading returns data...")
    try:
        returns_df = pd.read_csv(returns_file)
        returns_df['Date'] = pd.to_datetime(returns_df['Date'], utc=True)
        print(f"✅ Loaded {len(returns_df)} records")
        print(f"   Tickers: {returns_df['Ticker'].nunique()}")
        print(f"   Date range: {returns_df['Date'].min().date()} to {returns_df['Date'].max().date()}")
    except FileNotFoundError:
        print(f"❌ File not found: {returns_file}")
        return None

    # Step 2: Load and merge ESG data
    print("\n📂 Step 2: Loading and merging ESG data...")
    try:
        esg_df = pd.read_csv(esg_file)

        # Find ticker column in ESG data
        ticker_col_esg = None
        for col in ['Ticker', 'Symbol', 'ticker', 'symbol']:
            if col in esg_df.columns:
                ticker_col_esg = col
                break

        if ticker_col_esg:
            esg_df = esg_df.rename(columns={ticker_col_esg: 'Ticker'})

        print(f"✅ Loaded {len(esg_df)} companies")

        # Merge
        master_df = returns_df.merge(esg_df, on='Ticker', how='inner')
        print(f"   After merge: {len(master_df)} records, {master_df['Ticker'].nunique()} tickers")

        if len(master_df) == 0:
            print("❌ No matching tickers between returns and ESG data!")
            print(f"   Returns tickers sample: {returns_df['Ticker'].head().tolist()}")
            print(f"   ESG tickers sample: {esg_df['Ticker'].head().tolist()}")
            return None

    except FileNotFoundError:
        print(f"❌ File not found: {esg_file}")
        return None

    # Step 3: Load and merge risk-free rate
    print("\n📂 Step 3: Loading and merging risk-free rate...")
    try:
        rf_df = pd.read_csv(risk_free_file)
        rf_df['Date'] = pd.to_datetime(rf_df['Date'], utc=True)
        print(f"✅ Loaded {len(rf_df)} dates")

        # Merge
        master_df = master_df.merge(rf_df[['Date', 'Daily_RF_Rate']], on='Date', how='left')
        print(f"   After merge: {len(master_df)} records")

        # Check for missing risk-free rates
        missing_rf = master_df['Daily_RF_Rate'].isnull().sum()
        if missing_rf > 0:
            print(f"   ⚠️  Missing risk-free rate for {missing_rf} records")
            print(f"   Forward-filling missing values...")
            master_df = master_df.sort_values('Date')
            master_df['Daily_RF_Rate'] = master_df['Daily_RF_Rate'].ffill().bfill()

    except FileNotFoundError:
        print(f"⚠️  File not found: {risk_free_file}")
        print(f"   Setting risk-free rate to 0")
        master_df['Daily_RF_Rate'] = 0.0

    # Step 4: Load and merge market returns
    print("\n📂 Step 4: Loading and merging market returns...")
    try:
        market_df = pd.read_csv(market_returns_file)
        market_df['Date'] = pd.to_datetime(market_df['Date'], utc=True)
        print(f"✅ Loaded {len(market_df)} dates")

        # Merge
        master_df = master_df.merge(market_df[['Date', 'Market_Return']], on='Date', how='left')
        print(f"   After merge: {len(master_df)} records")

        # Check for missing market returns
        missing_mkt = master_df['Market_Return'].isnull().sum()
        if missing_mkt > 0:
            print(f"   ⚠️  Missing market return for {missing_mkt} records")
            print(f"   Forward-filling missing values...")
            master_df = master_df.sort_values('Date')
            master_df['Market_Return'] = master_df['Market_Return'].ffill().bfill()

    except FileNotFoundError:
        print(f"⚠️  File not found: {market_returns_file}")
        print(f"   Proceeding without market returns (will affect beta calculation)")
        master_df['Market_Return'] = np.nan

    # Step 5: Load and merge company info (market cap, sector)
    print("\n📂 Step 5: Loading and merging company information...")
    try:
        company_df = pd.read_csv(company_info_file)
        print(f"✅ Loaded {len(company_df)} companies")

        # Merge
        master_df = master_df.merge(
            company_df[['Ticker', 'Market_Cap', 'Sector', 'Industry']],
            on='Ticker',
            how='left'
        )
        print(f"   After merge: {len(master_df)} records")

        # Check for missing values
        missing_mcap = master_df['Market_Cap'].isnull().sum()
        missing_sector = master_df['Sector'].isnull().sum()

        if missing_mcap > 0:
            print(f"   ⚠️  Missing market cap for {missing_mcap} records")

        if missing_sector > 0:
            print(f"   ⚠️  Missing sector for {missing_sector} records")

    except FileNotFoundError:
        print(f"⚠️  File not found: {company_info_file}")
        print(f"   Proceeding without company info")
        master_df['Market_Cap'] = np.nan
        master_df['Sector'] = 'Unknown'
        master_df['Industry'] = 'Unknown'

    # Step 6: Calculate excess returns
    print("\n🔧 Calculating excess returns...")
    master_df['Excess_Return'] = master_df['Return'] - master_df['Daily_RF_Rate']

    # Step 7: Final validation and cleanup
    print("\n🔍 Final validation...")

    # Check for duplicates
    duplicates = master_df.duplicated(subset=['Ticker', 'Date']).sum()
    if duplicates > 0:
        print(f"   Found {duplicates} duplicate ticker-date pairs, removing...")
        master_df = master_df.drop_duplicates(subset=['Ticker', 'Date'], keep='first')

    # Check for infinite values
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(master_df[numeric_cols]).any(axis=1)
    inf_count = inf_mask.sum()
    if inf_count > 0:
        print(f"   Found {inf_count} rows with infinite values, removing...")
        master_df = master_df[~inf_mask]

    # Sort by ticker and date
    master_df = master_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Summary statistics
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(master_df)}")
    print(f"Unique tickers: {master_df['Ticker'].nunique()}")
    print(f"Date range: {master_df['Date'].min().date()} to {master_df['Date'].max().date()}")
    print(f"Trading days: {master_df['Date'].nunique()}")

    print("\nData completeness:")
    print(f"   Returns: {master_df['Return'].notna().sum()} / {len(master_df)}")
    print(f"   ESG scores: {master_df.filter(like='ESG').notna().all(axis=1).sum()} / {len(master_df)}")
    print(f"   Risk-free rate: {master_df['Daily_RF_Rate'].notna().sum()} / {len(master_df)}")
    print(f"   Market returns: {master_df['Market_Return'].notna().sum()} / {len(master_df)}")
    print(f"   Market cap: {master_df['Market_Cap'].notna().sum()} / {len(master_df)}")

    # Save master dataset
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_file, index=False)
    print(f"\n✅ Master dataset saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    print(master_df.head())

    print("\nColumn summary:")
    print(f"Total columns: {len(master_df.columns)}")
    print(f"Columns: {master_df.columns.tolist()}")

    return master_df


if __name__ == "__main__":
    result = merge_all_data()
    exit(0 if result is not None else 1)
