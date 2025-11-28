"""
Master script to process all raw data into analysis-ready format.

This script coordinates:
1. ESG data cleaning
2. Price data cleaning
3. Returns calculation (stock and market)
4. Risk-free rate processing
5. Data merging into master dataset

Usage:
    python scripts/process_data.py
"""
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.clean_esg import clean_esg_data
from src.data_processing.clean_prices import clean_price_data
from src.data_processing.calculate_returns import calculate_returns, calculate_market_returns
from src.data_processing.process_risk_free import process_risk_free_rate
from src.data_processing.merge_data import merge_all_data


def main():
    """
    Main function to orchestrate all data processing steps.
    """
    print("\n" + "=" * 60)
    print("ESG STOCK PERFORMANCE ANALYSIS - DATA PROCESSING")
    print("=" * 60)

    # Track success/failure
    results = {}

    # Step 1: Clean ESG data
    print("\n\n### STEP 1/6: Clean ESG Data ###")
    try:
        esg_df = clean_esg_data(
            input_file="data/raw/sp500_esg_data.csv",
            output_file="data/processed/esg_cleaned.csv"
        )
        results['esg'] = esg_df is not None
    except Exception as e:
        print(f"\n❌ Error cleaning ESG data: {e}")
        results['esg'] = False

    if not results['esg']:
        print("\n❌ ESG data cleaning failed. Cannot proceed.")
        return False

    # Step 2: Clean price data
    print("\n\n### STEP 2/6: Clean Price Data ###")
    try:
        prices_df = clean_price_data(
            input_file="data/raw/sp500_price_data.csv",
            output_file="data/processed/prices_cleaned.csv",
            start_date="2023-09-01",
            end_date="2024-08-31"
        )
        results['prices'] = prices_df is not None
    except Exception as e:
        print(f"\n❌ Error cleaning price data: {e}")
        results['prices'] = False

    if not results['prices']:
        print("\n❌ Price data cleaning failed. Cannot proceed.")
        return False

    # Step 3: Calculate returns
    print("\n\n### STEP 3/6: Calculate Returns ###")
    try:
        # Stock returns
        print("\n--- Stock Returns ---")
        returns_df = calculate_returns(
            input_file="data/processed/prices_cleaned.csv",
            output_file="data/processed/returns.csv",
            return_type="simple"
        )

        # Market returns
        print("\n--- Market Returns ---")
        market_df = calculate_market_returns(
            input_file="data/raw/sp500_index.csv",
            output_file="data/processed/market_returns.csv"
        )

        results['returns'] = returns_df is not None and market_df is not None
    except Exception as e:
        print(f"\n❌ Error calculating returns: {e}")
        results['returns'] = False

    if not results['returns']:
        print("\n❌ Returns calculation failed. Cannot proceed.")
        return False

    # Step 4: Process risk-free rate
    print("\n\n### STEP 4/6: Process Risk-Free Rate ###")
    try:
        rf_df = process_risk_free_rate(
            input_file="data/raw/DGS3MO.csv",
            output_file="data/processed/risk_free_rate.csv",
            trading_days_file="data/processed/returns.csv"
        )
        results['risk_free'] = rf_df is not None
    except Exception as e:
        print(f"\n⚠️  Error processing risk-free rate: {e}")
        print(f"   Will proceed with risk-free rate = 0")
        results['risk_free'] = False

    # Step 5: Merge all data
    print("\n\n### STEP 5/6: Merge All Datasets ###")
    try:
        master_df = merge_all_data(
            esg_file="data/processed/esg_cleaned.csv",
            returns_file="data/processed/returns.csv",
            market_returns_file="data/processed/market_returns.csv",
            risk_free_file="data/processed/risk_free_rate.csv",
            company_info_file="data/raw/company_info.csv",
            output_file="data/final/master_dataset.csv"
        )
        results['merge'] = master_df is not None
    except Exception as e:
        print(f"\n❌ Error merging data: {e}")
        results['merge'] = False

    if not results['merge']:
        print("\n❌ Data merging failed.")
        return False

    # Step 6: Data quality report
    print("\n\n### STEP 6/6: Data Quality Report ###")
    print("=" * 60)

    if master_df is not None:
        print("\n📊 Master Dataset Statistics:")
        print(f"   Total records: {len(master_df):,}")
        print(f"   Unique companies: {master_df['Ticker'].nunique()}")
        print(f"   Date range: {master_df['Date'].min().date()} to {master_df['Date'].max().date()}")
        print(f"   Trading days: {master_df['Date'].nunique()}")

        print("\n📈 Data Quality:")
        total_cells = len(master_df) * len(master_df.columns)
        missing_cells = master_df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        print(f"   Overall completeness: {completeness:.2f}%")

        # Missing data by column
        missing_by_col = master_df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

        if len(missing_by_col) > 0:
            print(f"\n   Columns with missing data:")
            for col, count in missing_by_col.head(10).items():
                pct = (count / len(master_df)) * 100
                print(f"      {col}: {count:,} ({pct:.1f}%)")
        else:
            print(f"\n   ✅ No missing data!")

    # Final summary
    print("\n\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    print("\nStatus:")
    print(f"   {'✅' if results['esg'] else '❌'} ESG Data Cleaning")
    print(f"   {'✅' if results['prices'] else '❌'} Price Data Cleaning")
    print(f"   {'✅' if results['returns'] else '❌'} Returns Calculation")
    print(f"   {'✅' if results['risk_free'] else '⚠️ '} Risk-Free Rate Processing")
    print(f"   {'✅' if results['merge'] else '❌'} Data Merging")

    success_count = sum([results['esg'], results['prices'], results['returns'], results['merge']])
    total_steps = 4  # Required steps (risk-free is optional)

    print(f"\nOverall: {success_count}/{total_steps} required steps completed")

    if success_count == total_steps:
        print("\n🎉 Data processing completed successfully!")
        print("\nFiles created:")
        print("   data/processed/esg_cleaned.csv")
        print("   data/processed/prices_cleaned.csv")
        print("   data/processed/returns.csv")
        print("   data/processed/market_returns.csv")
        if results['risk_free']:
            print("   data/processed/risk_free_rate.csv")
        print("   data/final/master_dataset.csv")

        print("\nNext steps:")
        print("   python scripts/run_analysis.py")
    else:
        print("\n❌ Data processing incomplete. Review error messages above.")

    print("=" * 60)

    return success_count == total_steps


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
