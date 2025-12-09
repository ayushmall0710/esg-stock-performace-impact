"""
Fetch company market cap and sector information from Yahoo Finance.
"""

import time
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


def fetch_company_info(
    tickers: List[str], output_dir: str = "data/raw", delay: float = 0.5
) -> bool:
    """
    Fetch market cap and sector data for a list of tickers from Yahoo Finance.

    Args:
        tickers: List of stock ticker symbols
        output_dir: Directory to save downloaded file
        delay: Delay between requests in seconds (to avoid rate limits)

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("Fetching Company Info from Yahoo Finance")
    print("=" * 60)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Fetching data for {len(tickers)} companies...")
    print(f"[NOTE]  Delay between requests: {delay}s")

    company_data = []
    successful = 0
    failed = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        try:
            # Progress indicator
            if i % 50 == 0:
                print(
                    f"\tProgress: {i}/{len(tickers)} ({successful} successful, {failed} failed)"
                )

            # Fetch ticker info
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract relevant fields
            company_info = {
                "Ticker": ticker,
                "Company_Name": info.get("longName", info.get("shortName", "Unknown")),
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "Market_Cap": info.get("marketCap", None),
                "Country": info.get("country", "Unknown"),
            }

            company_data.append(company_info)
            successful += 1

            # Delay to avoid rate limiting
            time.sleep(delay)

        except Exception as e:
            print(f"\t[WARNING]  Failed to fetch data for {ticker}: {str(e)[:50]}")
            failed += 1
            failed_tickers.append(ticker)

            # Add placeholder data for failed tickers
            company_data.append(
                {
                    "Ticker": ticker,
                    "Company_Name": "Unknown",
                    "Sector": "Unknown",
                    "Industry": "Unknown",
                    "Market_Cap": None,
                    "Country": "Unknown",
                }
            )

    # Create DataFrame
    df = pd.DataFrame(company_data)

    # Save to CSV
    output_file = Path(output_dir) / "company_info.csv"
    df.to_csv(output_file, index=False)

    print(f"\n[OK] Data saved to: {output_file}")
    print("\n[INFO] Summary:")
    print(f"\tTotal companies: {len(tickers)}")
    print(f"\tSuccessfully fetched: {successful}")
    print(f"\tFailed: {failed}")

    if failed_tickers:
        print(f"\n[WARNING]  Failed tickers ({len(failed_tickers)}):")
        print(f"\t{', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"\t... and {len(failed_tickers) - 10} more")

    # Show sector distribution
    print("\n[STATS] Sector distribution:")
    sector_counts = df["Sector"].value_counts()
    for sector, count in sector_counts.head(10).items():
        print(f"\t{sector}: {count}")

    # Show market cap statistics
    if df["Market_Cap"].notna().sum() > 0:
        print("\n💰 Market Cap statistics (in billions):")
        market_caps = df["Market_Cap"].dropna() / 1e9
        print(f"\tMean: ${market_caps.mean():.2f}B")
        print(f"\tMedian: ${market_caps.median():.2f}B")
        print(f"\tMin: ${market_caps.min():.2f}B")
        print(f"\tMax: ${market_caps.max():.2f}B")
    else:
        print("\n[WARNING]  No market cap data available")

    return True


def load_tickers_from_esg_data(
    esg_file: str = "data/raw/sp500_esg_data.csv",
) -> List[str]:
    """
    Load ticker symbols from ESG data file.

    Args:
        esg_file: Path to ESG data CSV file

    Returns:
        List of ticker symbols
    """
    print(f"\n[LOADING] Loading tickers from: {esg_file}")

    try:
        df = pd.read_csv(esg_file)

        # Try common column names for tickers
        ticker_columns = ["Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"]
        ticker_col = None

        for col in ticker_columns:
            if col in df.columns:
                ticker_col = col
                break

        if ticker_col is None:
            print(
                f"[ERROR] Could not find ticker column. Available columns: {df.columns.tolist()}"
            )
            return []

        tickers = df[ticker_col].dropna().unique().tolist()
        print(f"[OK] Found {len(tickers)} unique tickers")

        return tickers

    except FileNotFoundError:
        print(f"[ERROR] File not found: {esg_file}")
        print("Please download the Kaggle dataset first using fetch_kaggle.py")
        return []
    except Exception as e:
        print(f"[ERROR] Error loading tickers: {e}")
        return []


if __name__ == "__main__":
    # Load tickers from ESG data
    tickers = load_tickers_from_esg_data()

    if not tickers:
        print("\n[ERROR] No tickers found. Please ensure ESG data is downloaded first.")
        print("Run: python src/data_acquisition/fetch_kaggle.py")
        exit(1)

    # Fetch company info
    success = fetch_company_info(tickers)
    exit(0 if success else 1)
