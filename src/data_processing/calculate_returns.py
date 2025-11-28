"""
Calculate daily returns from stock prices.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def calculate_returns(input_file: str = "data/processed/prices_cleaned.csv",
                      output_file: str = "data/processed/returns.csv",
                      return_type: str = "simple") -> pd.DataFrame:
    """
    Calculate daily returns from price data.

    Args:
        input_file: Path to cleaned price data
        output_file: Path to save returns data
        return_type: Type of returns ('simple' or 'log')

    Returns:
        DataFrame with returns
    """
    print("=" * 60)
    print("Calculating Daily Returns")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please run price cleaning script first.")
        return None

    # Identify columns
    date_col = None
    for col in ['Date', 'date', 'DATE']:
        if col in df.columns:
            date_col = col
            break

    ticker_col = None
    for col in ['Ticker', 'Symbol', 'ticker', 'symbol']:
        if col in df.columns:
            ticker_col = col
            break

    price_col = None
    for col in ['Close', 'Adj Close', 'close', 'adj_close', 'Price']:
        if col in df.columns:
            price_col = col
            break

    if not all([date_col, ticker_col, price_col]):
        print(f"\n❌ Could not identify required columns")
        print(f"   Date column: {date_col}")
        print(f"   Ticker column: {ticker_col}")
        print(f"   Price column: {price_col}")
        return None

    print(f"\n✅ Using columns:")
    print(f"   Date: {date_col}")
    print(f"   Ticker: {ticker_col}")
    print(f"   Price: {price_col}")

    # Parse dates and sort
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # Calculate returns by ticker
    print(f"\n🔧 Calculating {return_type} returns...")

    if return_type == "simple":
        # Simple returns: (P_t - P_t-1) / P_t-1
        df['Return'] = df.groupby(ticker_col)[price_col].pct_change()
    elif return_type == "log":
        # Log returns: ln(P_t / P_t-1)
        df['Return'] = df.groupby(ticker_col)[price_col].transform(
            lambda x: np.log(x / x.shift(1))
        )
    else:
        print(f"❌ Unknown return type: {return_type}")
        print("   Use 'simple' or 'log'")
        return None

    # Remove first row for each ticker (NaN from pct_change)
    print("🔧 Removing NaN values from first day...")
    initial_count = len(df)
    df = df.dropna(subset=['Return'])
    print(f"   Removed {initial_count - len(df)} rows")

    # Check for infinite values
    print("\n🔍 Checking for problematic returns...")
    inf_count = np.isinf(df['Return']).sum()
    if inf_count > 0:
        print(f"   Found {inf_count} infinite values, removing")
        df = df[~np.isinf(df['Return'])]

    # Flag extreme returns (> 100% or < -100% for simple returns)
    if return_type == "simple":
        extreme_mask = (df['Return'] > 1.0) | (df['Return'] < -1.0)
        extreme_count = extreme_mask.sum()
        if extreme_count > 0:
            print(f"   Found {extreme_count} extreme returns (>100% or <-100%)")
            print(f"   Max return: {df['Return'].max():.2%}")
            print(f"   Min return: {df['Return'].min():.2%}")
            print(f"   These may indicate stock splits or data errors")

    # Return statistics
    print("\n📊 Return Statistics:")
    print(f"   Mean daily return: {df['Return'].mean():.4%}")
    print(f"   Median daily return: {df['Return'].median():.4%}")
    print(f"   Std dev: {df['Return'].std():.4%}")
    print(f"   Min: {df['Return'].min():.2%}")
    print(f"   Max: {df['Return'].max():.2%}")

    # Check return distribution
    print("\n📈 Return Distribution:")
    print(df['Return'].describe())

    # Save returns data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Returns data saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    display_cols = [date_col, ticker_col, price_col, 'Return']
    print(df[display_cols].head(10))

    print("\n" + "=" * 60)
    print("CALCULATION SUMMARY")
    print("=" * 60)
    print(f"Total return records: {len(df)}")
    print(f"Tickers: {df[ticker_col].nunique()}")
    print(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
    print(f"Return type: {return_type}")

    return df


def calculate_market_returns(input_file: str = "data/raw/sp500_index.csv",
                              output_file: str = "data/processed/market_returns.csv") -> pd.DataFrame:
    """
    Calculate daily returns for market index (S&P 500).

    Args:
        input_file: Path to S&P 500 index data
        output_file: Path to save market returns

    Returns:
        DataFrame with market returns
    """
    print("\n" + "=" * 60)
    print("Calculating Market Returns (S&P 500)")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)

        # Handle multi-level columns if present (from yfinance)
        if len(df.columns) > 1 and df.iloc[0].astype(str).str.contains('^GSPC', na=False).any():
            # Skip the ticker row if it exists
            df = df.iloc[1:]
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"✅ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please run data download script first.")
        return None

    # Use Close price for returns
    if 'Close' not in df.columns:
        print(f"❌ 'Close' column not found in: {df.columns.tolist()}")
        return None

    # Calculate simple returns
    df['Market_Return'] = df['Close'].pct_change(fill_method=None)

    # Remove first NaN
    df = df.dropna(subset=['Market_Return'])

    # Statistics
    print("\n📊 Market Return Statistics:")
    print(f"   Mean daily return: {df['Market_Return'].mean():.4%}")
    print(f"   Std dev: {df['Market_Return'].std():.4%}")
    print(f"   Annualized return: {(1 + df['Market_Return'].mean()) ** 252 - 1:.2%}")
    print(f"   Annualized volatility: {df['Market_Return'].std() * np.sqrt(252):.2%}")

    # Save
    df = df.reset_index()
    df = df.rename(columns={'Date': 'Date'})
    df[['Date', 'Market_Return']].to_csv(output_file, index=False)
    print(f"\n✅ Market returns saved to: {output_file}")

    return df


if __name__ == "__main__":
    # Calculate stock returns
    returns = calculate_returns()

    # Calculate market returns
    market_returns = calculate_market_returns()

    success = returns is not None and market_returns is not None
    exit(0 if success else 1)
