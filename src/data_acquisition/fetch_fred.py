"""
Fetch U.S. 3-Month Treasury rate data from FRED.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()


def download_fred_data(output_dir: str = "data/raw", start_date: str = "2023-09-01", end_date: str = "2024-08-31") -> bool:
    """
    Download 3-Month Treasury rate (DGS3MO) from FRED.

    Args:
        output_dir: Directory to save downloaded file
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("Downloading U.S. 3-Month Treasury Rate from FRED")
    print("=" * 60)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check for FRED API key
    fred_api_key = os.getenv('FRED_API_KEY')

    if fred_api_key and fred_api_key != 'your_fred_api_key_here':
        # Option A: Use FRED API via pandas_datareader
        print("\n📡 Using FRED API to fetch data...")
        try:
            from pandas_datareader import data as pdr

            print(f"📅 Date range: {start_date} to {end_date}")

            # Fetch data from FRED
            df = pdr.DataReader('DGS3MO', 'fred', start_date, end_date)

            # Save to CSV
            output_file = Path(output_dir) / "DGS3MO.csv"
            df.to_csv(output_file)

            print(f"\n✅ Data saved to: {output_file}")
            print(f"📊 Records downloaded: {len(df)}")
            print(f"📈 Date range: {df.index.min()} to {df.index.max()}")

            # Show summary
            print("\nData preview:")
            print(df.head())
            print(f"\nMissing values: {df.isnull().sum().values[0]}")

            return True

        except ImportError:
            print("\n❌ pandas_datareader not installed!")
            print("Install it with: pip install pandas-datareader")
            print("\nFalling back to manual download instructions...")
            fred_api_key = None

        except Exception as e:
            print(f"\n❌ Error fetching data from FRED API: {e}")
            print("\nFalling back to manual download instructions...")
            fred_api_key = None

    # Option B: Manual download instructions
    if not fred_api_key or fred_api_key == 'your_fred_api_key_here':
        print("\n📋 FRED API key not configured. Manual download required.")
        print("\nOption 1 - Get a free FRED API key (recommended):")
        print("   1. Sign up at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   2. Add your API key to .env file:")
        print("      FRED_API_KEY=your_actual_api_key")
        print("   3. Run this script again")

        print("\nOption 2 - Manual CSV download:")
        print("   1. Go to: https://fred.stlouisfed.org/series/DGS3MO")
        print("   2. Click 'Download' button")
        print("   3. Select date range: 2023-09-01 to 2024-08-31")
        print("   4. Download as CSV")
        print(f"   5. Save as: {Path(output_dir) / 'DGS3MO.csv'}")

        # Check if file already exists from manual download
        output_file = Path(output_dir) / "DGS3MO.csv"
        if output_file.exists():
            print(f"\n✅ Found existing file: {output_file}")
            try:
                df = pd.read_csv(output_file, index_col=0, parse_dates=True)
                print(f"📊 Records: {len(df)}")
                print(f"📈 Date range: {df.index.min()} to {df.index.max()}")
                return True
            except Exception as e:
                print(f"⚠️  Error reading file: {e}")
                return False
        else:
            print(f"\n⚠️  File not found: {output_file}")
            print("Please complete manual download or set up FRED API key.")
            return False


if __name__ == "__main__":
    success = download_fred_data()
    exit(0 if success else 1)
