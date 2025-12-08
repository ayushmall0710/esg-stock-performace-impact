"""
Fetch S&P 500 index data from Yahoo Finance.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_sp500_index(
    output_dir: str = "data/raw",
    start_date: str = "2023-09-01",
    end_date: str = "2024-08-31",
) -> bool:
    """
    Download S&P 500 index daily prices from Yahoo Finance.

    Args:
        output_dir: Directory to save downloaded file
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("Downloading S&P 500 Index Data from Yahoo Finance")
    print("=" * 60)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n[FETCHING] Fetching S&P 500 (^GSPC) data...")
    print(f"[DATE] Date range: {start_date} to {end_date}")

    try:
        # Download S&P 500 index data
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)

        if sp500.empty:
            print("\n[ERROR] No data downloaded. Please check your internet connection.")
            return False

        # Save to CSV with proper formatting
        # yfinance returns multi-level columns, flatten them
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)

        output_file = Path(output_dir) / "sp500_index.csv"
        sp500.to_csv(output_file)

        print(f"\n[OK] Data saved to: {output_file}")
        print(f"[INFO] Trading days: {len(sp500)}")
        print(
            f"[STATS] Date range: {sp500.index.min().date()} to {sp500.index.max().date()}"
        )

        # Show summary statistics
        print("\nPrice summary:")
        open_min = sp500["Open"].min()
        open_max = sp500["Open"].max()
        close_min = sp500["Close"].min()
        close_max = sp500["Close"].max()
        print(f"\tOpening price range: ${open_min:.2f} - ${open_max:.2f}")
        print(f"\tClosing price range: ${close_min:.2f} - ${close_max:.2f}")

        # Preview data
        print("\nData preview:")
        print(sp500.head())

        return True

    except Exception as e:
        print(f"\n[ERROR] Error downloading S&P 500 data: {e}")
        print("\nTroubleshooting:")
        print("\t- Check your internet connection")
        print("\t- Verify yfinance package is installed: pip install yfinance")
        print("\t- Try again in a few minutes (Yahoo Finance may have rate limits)")
        return False


if __name__ == "__main__":
    success = download_sp500_index()
    exit(0 if success else 1)
