"""
Process risk-free rate data from FRED.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def process_risk_free_rate(input_file: str = "data/raw/DGS3MO.csv",
                            output_file: str = "data/processed/risk_free_rate.csv",
                            trading_days_file: str = "data/processed/returns.csv") -> pd.DataFrame:
    """
    Process risk-free rate data:
    1. Convert annual 3-month T-bill rate to daily rate
    2. Align with trading days from stock data
    3. Forward-fill for missing trading days

    Args:
        input_file: Path to FRED DGS3MO data
        output_file: Path to save processed risk-free rate
        trading_days_file: Path to stock returns (to get trading days)

    Returns:
        DataFrame with daily risk-free rates
    """
    print("=" * 60)
    print("Processing Risk-Free Rate Data")
    print("=" * 60)

    # Load FRED data
    print(f"\n📂 Loading data from: {input_file}")
    try:
        # FRED data typically has 'DATE' and value column
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please run data download script first.")
        return None

    print(f"\nColumns: {df.columns.tolist()}")

    # Identify date column
    date_col = None
    for col in ['DATE', 'Date', 'date']:
        if col in df.columns:
            date_col = col
            break

    # Identify rate column
    rate_col = None
    for col in ['DGS3MO', 'VALUE', 'Value', 'value']:
        if col in df.columns:
            rate_col = col
            break

    if not date_col or not rate_col:
        # Try using first two columns
        if len(df.columns) >= 2:
            date_col = df.columns[0]
            rate_col = df.columns[1]
            print(f"⚠️  Guessing columns: {date_col} (date), {rate_col} (rate)")
        else:
            print(f"❌ Could not identify date and rate columns")
            return None

    print(f"\n✅ Using columns:")
    print(f"   Date: {date_col}")
    print(f"   Rate: {rate_col}")

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.rename(columns={date_col: 'Date', rate_col: 'Annual_Rate'})

    # Handle missing values (marked as '.' in FRED data sometimes)
    if df['Annual_Rate'].dtype == object:
        df['Annual_Rate'] = pd.to_numeric(df['Annual_Rate'], errors='coerce')

    # Drop NaN values
    print(f"\n🔧 Handling missing values...")
    missing_count = df['Annual_Rate'].isnull().sum()
    if missing_count > 0:
        print(f"   Found {missing_count} missing values")
        print(f"   Forward-filling gaps...")
        df['Annual_Rate'] = df['Annual_Rate'].ffill().bfill()

    # Convert annual rate to daily rate
    # Formula: daily_rate = (1 + annual_rate/100) ^ (1/252) - 1
    print(f"\n🔧 Converting annual rate to daily rate...")
    print(f"   Assuming 252 trading days per year")

    df['Daily_RF_Rate'] = (1 + df['Annual_Rate'] / 100) ** (1 / 252) - 1

    # Load trading days from stock data
    print(f"\n📂 Loading trading days from: {trading_days_file}")
    try:
        returns_df = pd.read_csv(trading_days_file)
        date_col_returns = None
        for col in ['Date', 'date', 'DATE']:
            if col in returns_df.columns:
                date_col_returns = col
                break

        if date_col_returns:
            returns_df[date_col_returns] = pd.to_datetime(returns_df[date_col_returns], utc=True)
            trading_days = returns_df[date_col_returns].unique()
            trading_days = pd.Series(trading_days).sort_values()
            print(f"✅ Found {len(trading_days)} unique trading days")
        else:
            print(f"⚠️  Could not find date column in returns data")
            trading_days = None
    except FileNotFoundError:
        print(f"⚠️  Trading days file not found, using all dates from FRED")
        trading_days = None

    # Align with trading days
    if trading_days is not None:
        print(f"\n🔧 Aligning risk-free rate with trading days...")
        trading_df = pd.DataFrame({'Date': trading_days})
        df = trading_df.merge(df, on='Date', how='left')

        # Forward-fill for trading days that don't have FRED data
        df['Daily_RF_Rate'] = df['Daily_RF_Rate'].ffill()
        df['Annual_Rate'] = df['Annual_Rate'].ffill()

        print(f"   Aligned to {len(df)} trading days")

    # Statistics
    print("\n📊 Risk-Free Rate Statistics:")
    print(f"   Annual rate range: {df['Annual_Rate'].min():.4f}% - {df['Annual_Rate'].max():.4f}%")
    print(f"   Mean annual rate: {df['Annual_Rate'].mean():.4f}%")
    print(f"   Mean daily rate: {df['Daily_RF_Rate'].mean():.6f} ({df['Daily_RF_Rate'].mean() * 100:.4f}%)")

    # Annualized daily rate (for verification)
    annualized_from_daily = (1 + df['Daily_RF_Rate'].mean()) ** 252 - 1
    print(f"   Annualized daily rate: {annualized_from_daily * 100:.4f}% (should match annual rate)")

    # Check for any remaining NaN
    remaining_nan = df['Daily_RF_Rate'].isnull().sum()
    if remaining_nan > 0:
        print(f"\n⚠️  Warning: {remaining_nan} dates still have NaN values")
        print(f"   Dropping these dates...")
        df = df.dropna(subset=['Daily_RF_Rate'])

    # Save processed data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df[['Date', 'Daily_RF_Rate', 'Annual_Rate']].to_csv(output_file, index=False)
    print(f"\n✅ Processed data saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    print(df[['Date', 'Annual_Rate', 'Daily_RF_Rate']].head(10))

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total days: {len(df)}")
    print(f"Average annual rate: {df['Annual_Rate'].mean():.4f}%")
    print(f"Average daily rate: {df['Daily_RF_Rate'].mean() * 100:.4f}%")

    return df


if __name__ == "__main__":
    result = process_risk_free_rate()
    exit(0 if result is not None else 1)
