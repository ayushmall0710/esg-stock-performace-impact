"""
Master script to download all required data for ESG Stock Performance Analysis.

This script coordinates the download of:
1. S&P 500 ESG and stock price data from Kaggle
2. U.S. 3-Month Treasury rate from FRED
3. S&P 500 index data from Yahoo Finance
4. Company market cap and sector information from Yahoo Finance

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"[INFO] Loaded environment variables from {env_file}")
else:
    print(f"[INFO] No .env file found at {env_file}")

from src.data_acquisition.fetch_company_info import (
    fetch_company_info,
    load_tickers_from_esg_data,
)
from src.data_acquisition.fetch_fred import download_fred_data
from src.data_acquisition.fetch_kaggle import download_kaggle_dataset
from src.data_acquisition.fetch_market_data import download_sp500_index


def main():
    """
    Main function to orchestrate all data downloads.
    """
    print("\n" + "=" * 60)
    print("ESG STOCK PERFORMANCE ANALYSIS - DATA DOWNLOAD")
    print("=" * 60)

    # Track success/failure
    results = {}

    # Step 1: Download Kaggle dataset (ESG scores and stock prices)
    print("\n\n### STEP 1/4: Kaggle Dataset ###")
    results["kaggle"] = download_kaggle_dataset(output_dir="data/raw")

    if not results["kaggle"]:
        print(
            "\n[WARNING] Kaggle download failed. Please set up Kaggle API credentials and try again."
        )
        print("The remaining data sources will still be attempted.\n")

    # Step 2: Download FRED data (risk-free rate)
    print("\n\n### STEP 2/4: FRED Treasury Rate ###")
    results["fred"] = download_fred_data(
        output_dir="data/raw", start_date="2023-09-01", end_date="2024-08-31"
    )

    if not results["fred"]:
        print(
            "\n[WARNING] FRED download incomplete. You may need to set up FRED API key or download manually."
        )
        print("See instructions above.\n")

    # Step 3: Download S&P 500 index data
    print("\n\n### STEP 3/4: S&P 500 Index Data ###")
    results["sp500_index"] = download_sp500_index(
        output_dir="data/raw", start_date="2023-09-01", end_date="2024-08-31"
    )

    if not results["sp500_index"]:
        print(
            "\n[WARNING] S&P 500 index download failed. Please check your internet connection.\n"
        )

    # Step 4: Download company information (requires Kaggle data first)
    print("\n\n### STEP 4/4: Company Information ###")
    if results["kaggle"]:
        # Load tickers from downloaded ESG data
        tickers = load_tickers_from_esg_data(esg_file="data/raw/sp500_esg_data.csv")

        if tickers:
            print(f"\n[INFO] Fetching company info for {len(tickers)} tickers...")
            print("[NOTE] This may take several minutes due to rate limiting...")
            results["company_info"] = fetch_company_info(
                tickers=tickers,
                output_dir="data/raw",
                delay=0.5,  # 0.5 second delay between requests
            )
        else:
            print("\n[ERROR] Could not load tickers from ESG data.")
            results["company_info"] = False
    else:
        print("\n[WARNING] Skipping company info download (requires Kaggle data first)")
        results["company_info"] = False

    # Final summary
    print("\n\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print("\nStatus:")
    print(
        f"\t{'[OK]' if results['kaggle'] else '[FAILED]'} Kaggle Dataset (ESG & Prices)"
    )
    print(f"\t{'[OK]' if results['fred'] else '[FAILED]'} FRED Treasury Rate")
    print(f"\t{'[OK]' if results['sp500_index'] else '[OK]'} S&P 500 Index")
    print(f"\t{'[OK]' if results['company_info'] else '[FAILED]'} Company Information")

    total_success = sum(results.values())
    total_tasks = len(results)

    print(
        f"\nOverall: {total_success}/{total_tasks} data sources downloaded successfully"
    )

    if total_success == total_tasks:
        print("\n[SUCCESS] All data downloaded successfully!")
        print("\nNext steps:")
        print("\tpython scripts/process_data.py")
    elif total_success > 0:
        print(
            "\n[WARNING] Some data sources failed. Review messages above for details."
        )
        print(
            "You may still proceed with available data, but some analyses may be limited."
        )
    else:
        print("\n[ERROR] All downloads failed. Please check:")
        print("\t- Internet connection")
        print("\t- Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("\t- FRED API key (.env file)")

    print("\nData files saved to: data/raw/")
    print("=" * 60)

    # Return success if at least Kaggle data was downloaded (minimum requirement)
    return results["kaggle"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
