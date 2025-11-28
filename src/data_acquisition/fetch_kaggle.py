"""
Fetch S&P 500 ESG and stock price data from Kaggle.
"""
import os
import subprocess
from pathlib import Path


def download_kaggle_dataset(output_dir: str = "data/raw") -> bool:
    """
    Download S&P 500 ESG and Stocks dataset from Kaggle.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("Downloading S&P 500 ESG and Stocks Data from Kaggle")
    print("=" * 60)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if Kaggle credentials are set up
    # Kaggle API supports both environment variables and kaggle.json file
    has_env_creds = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    has_json_creds = kaggle_json.exists()

    if not has_env_creds and not has_json_creds:
        print("\n❌ Kaggle API credentials not found!")
        print("\nPlease set up your Kaggle API credentials using one of these methods:")
        print("\nMethod 1: Environment Variables (recommended for this project)")
        print("1. Copy .env.template to .env")
        print("2. Edit .env and add your Kaggle username and API key")
        print("3. Make sure to load the .env file before running this script")
        print("\nMethod 2: Kaggle JSON file")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Run the following commands:")
        print(f"   mkdir -p {kaggle_dir}")
        print(f"   mv ~/Downloads/kaggle.json {kaggle_dir}/")
        print(f"   chmod 600 {kaggle_json}")
        print("\nThen run this script again.")
        return False

    if has_env_creds:
        print(f"\n🔑 Using Kaggle credentials from environment variables")
    else:
        print(f"\n🔑 Using Kaggle credentials from {kaggle_json}")

    # Dataset identifier
    dataset = "rikinzala/s-and-p-500-esg-and-stocks-data-2023-24"

    print(f"\n📥 Downloading dataset: {dataset}")
    print(f"📁 Saving to: {output_dir}\n")

    try:
        # Download and unzip the dataset
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", output_dir, "--unzip"],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        # Verify downloaded files
        expected_files = ["sp500_esg_data.csv", "sp500_price_data.csv"]
        downloaded_files = []

        for file in expected_files:
            file_path = Path(output_dir) / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"✅ Found: {file} ({size_mb:.2f} MB)")
                downloaded_files.append(file)
            else:
                print(f"❌ Missing: {file}")

        if len(downloaded_files) == len(expected_files):
            print("\n✅ Kaggle dataset downloaded successfully!")
            return True
        else:
            print(f"\n⚠️  Warning: Expected {len(expected_files)} files, found {len(downloaded_files)}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("\n❌ Kaggle CLI not found!")
        print("Please install the Kaggle package:")
        print("   pip install kaggle")
        return False


if __name__ == "__main__":
    success = download_kaggle_dataset()
    exit(0 if success else 1)
