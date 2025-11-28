"""
Clean and process stock price data.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def clean_price_data(input_file: str = "data/raw/sp500_price_data.csv",
                     output_file: str = "data/processed/prices_cleaned.csv",
                     start_date: str = "2023-09-01",
                     end_date: str = "2024-08-31") -> pd.DataFrame:
    """
    Clean and process stock price data.

    Steps:
    1. Parse date column as datetime
    2. Filter to analysis window
    3. Handle missing prices (forward-fill with limits)
    4. Drop tickers with > 10% missing data
    5. Ensure proper chronological order

    Args:
        input_file: Path to raw price data
        output_file: Path to save cleaned data
        start_date: Start of analysis window
        end_date: End of analysis window

    Returns:
        Cleaned DataFrame
    """
    print("=" * 60)
    print("Cleaning Price Data")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please run data download script first.")
        return None

    print(f"\nColumns: {df.columns.tolist()}")

    # Identify date column
    date_candidates = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp']
    date_col = None
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        print(f"\n❌ Could not find date column in: {df.columns.tolist()}")
        return None

    # Parse dates
    print(f"\n🔧 Parsing dates from column: '{date_col}'")
    df[date_col] = pd.to_datetime(df[date_col])

    # Filter to analysis window
    print(f"🔧 Filtering to date range: {start_date} to {end_date}")
    original_count = len(df)
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
    print(f"   Kept {len(df)} / {original_count} records")

    # Sort by date
    print("🔧 Sorting by date...")
    df = df.sort_values(date_col).reset_index(drop=True)

    # Identify price columns (typically: Open, High, Low, Close, Volume)
    price_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(price_type in col_lower for price_type in ['open', 'high', 'low', 'close', 'adj', 'price']):
            if col != date_col:
                price_columns.append(col)

    print(f"\n📊 Identified price columns: {price_columns}")

    # Check if data is in wide format (one row per date, multiple ticker columns)
    # or long format (one row per ticker-date pair)
    if 'Ticker' in df.columns or 'Symbol' in df.columns:
        # Long format
        print("\n📋 Data format: Long (one row per ticker-date)")
        ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'

        # Analyze missing data by ticker
        tickers = df[ticker_col].unique()
        print(f"\n🔍 Analyzing {len(tickers)} unique tickers...")

        ticker_stats = []
        for ticker in tickers:
            ticker_df = df[df[ticker_col] == ticker]
            expected_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            actual_days = len(ticker_df)
            missing_pct = (expected_days - actual_days) / expected_days * 100

            # Check for NaN values in price columns
            nan_counts = ticker_df[price_columns].isnull().sum().sum()

            ticker_stats.append({
                'Ticker': ticker,
                'Days': actual_days,
                'Expected': expected_days,
                'Missing_Pct': missing_pct,
                'NaN_Count': nan_counts
            })

        stats_df = pd.DataFrame(ticker_stats)

        # Drop tickers with > 10% missing data
        threshold = 10
        bad_tickers = stats_df[stats_df['Missing_Pct'] > threshold]['Ticker'].tolist()

        if bad_tickers:
            print(f"\n⚠️  Dropping {len(bad_tickers)} tickers with > {threshold}% missing data:")
            print(f"   {', '.join(bad_tickers[:10])}")
            if len(bad_tickers) > 10:
                print(f"   ... and {len(bad_tickers) - 10} more")

            df = df[~df[ticker_col].isin(bad_tickers)].copy()

        # Handle remaining missing values with forward fill (max 5 days)
        print(f"\n🔧 Forward-filling missing values (max 5 days)...")
        df = df.sort_values([ticker_col, date_col])

        for col in price_columns:
            df[col] = df.groupby(ticker_col)[col].transform(
                lambda x: x.ffill(limit=5)
            )

        # Drop any remaining NaN values
        remaining_na = df[price_columns].isnull().sum().sum()
        if remaining_na > 0:
            print(f"   Dropping {remaining_na} remaining NaN values")
            df = df.dropna(subset=price_columns)

    else:
        # Wide format
        print("\n📋 Data format: Wide (one row per date, columns per ticker)")
        print("   Converting to long format...")

        # Set date as index
        df = df.set_index(date_col)

        # Forward fill missing values (max 5 days for weekends/holidays)
        df = df.ffill(limit=5)

        # Drop columns (tickers) with > 10% missing data
        threshold = 0.1
        missing_pct = df.isnull().sum() / len(df)
        bad_columns = missing_pct[missing_pct > threshold].index.tolist()

        if bad_columns:
            print(f"\n⚠️  Dropping {len(bad_columns)} tickers with > 10% missing data")
            df = df.drop(columns=bad_columns)

        # Check for infinite values before conversion
        print("\n🔍 Checking for infinite values...")
        inf_count = np.isinf(df.values).sum()
        if inf_count > 0:
            print(f"   Found {inf_count} infinite values, replacing with NaN")
            df = df.replace([np.inf, -np.inf], np.nan)

        # Drop any remaining NaN values
        df = df.dropna()

        # Convert wide to long format for consistency
        df = df.reset_index().melt(id_vars=[date_col], var_name='Ticker', value_name='Close')
        ticker_col = 'Ticker'
        price_columns = ['Close']

    # Final statistics
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
    print(f"Trading days: {df[date_col].nunique()}")
    print(f"Tickers: {df[ticker_col].nunique()}")
    print(f"Total records: {len(df)}")

    # Save cleaned data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned data saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    print(df.head(10))

    return df


if __name__ == "__main__":
    result = clean_price_data()
    exit(0 if result is not None else 1)
