"""
Master script to run feature engineering on processed data.

This script:
1. Calculates performance metrics (Sharpe ratio, returns)
2. Calculates risk metrics (volatility, beta)
3. Creates control variables (log market cap, sector dummies)
4. Aggregates all features into firm-level analysis dataset

Usage:
    python scripts/run_feature_engineering.py
"""
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.feature_engineering.performance_metrics import calculate_performance_metrics
from src.feature_engineering.risk_metrics import calculate_risk_metrics
from src.feature_engineering.controls import create_control_variables
from src.feature_engineering.aggregate_features import aggregate_all_features


def main():
    """
    Main function to orchestrate feature engineering.
    """
    print("\n" + "=" * 60)
    print("ESG STOCK PERFORMANCE ANALYSIS - FEATURE ENGINEERING")
    print("=" * 60)

    # Load master dataset
    print("\n### Loading Master Dataset ###")
    master_file = "data/final/master_dataset.csv"

    try:
        master_df = pd.read_csv(master_file)
        print(f"✅ Loaded {len(master_df)} records from {master_file}")
        print(f"   Tickers: {master_df['Ticker'].nunique()}")
        print(f"   Date range: {master_df['Date'].min()} to {master_df['Date'].max()}")
    except FileNotFoundError:
        print(f"❌ File not found: {master_file}")
        print("Please run data processing first: python scripts/process_data.py")
        return False

    # Step 1: Calculate performance metrics
    print("\n\n### STEP 1/4: Performance Metrics ###")
    try:
        performance_df = calculate_performance_metrics(
            df=master_df,
            ticker_col='Ticker',
            return_col='Return',
            excess_return_col='Excess_Return'
        )

        # Save intermediate results
        perf_file = "data/processed/performance_metrics.csv"
        performance_df.to_csv(perf_file, index=False)
        print(f"\n💾 Performance metrics saved to: {perf_file}")

    except Exception as e:
        print(f"\n❌ Error calculating performance metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Calculate risk metrics
    print("\n\n### STEP 2/4: Risk Metrics ###")
    try:
        risk_df = calculate_risk_metrics(
            df=master_df,
            ticker_col='Ticker',
            return_col='Return',
            excess_return_col='Excess_Return',
            market_return_col='Market_Return'
        )

        # Save intermediate results
        risk_file = "data/processed/risk_metrics.csv"
        risk_df.to_csv(risk_file, index=False)
        print(f"\n💾 Risk metrics saved to: {risk_file}")

    except Exception as e:
        print(f"\n❌ Error calculating risk metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Create control variables
    print("\n\n### STEP 3/4: Control Variables ###")
    try:
        # Get unique tickers with company info
        company_df = master_df[['Ticker', 'Market_Cap', 'Sector']].drop_duplicates(subset=['Ticker'])

        controls_df = create_control_variables(
            df=company_df,
            ticker_col='Ticker',
            market_cap_col='Market_Cap',
            sector_col='Sector'
        )

        # Save intermediate results
        controls_file = "data/processed/control_variables.csv"
        controls_df.to_csv(controls_file, index=False)
        print(f"\n💾 Control variables saved to: {controls_file}")

    except Exception as e:
        print(f"\n❌ Error creating control variables: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Aggregate all features
    print("\n\n### STEP 4/4: Aggregate All Features ###")
    try:
        analysis_df = aggregate_all_features(
            master_file=master_file,
            performance_df=performance_df,
            risk_df=risk_df,
            controls_df=controls_df,
            output_file="data/final/analysis_dataset.csv"
        )

    except Exception as e:
        print(f"\n❌ Error aggregating features: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Final summary
    print("\n\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    print(f"\n✅ All steps completed successfully!")

    print(f"\nFiles created:")
    print(f"   data/processed/performance_metrics.csv")
    print(f"   data/processed/risk_metrics.csv")
    print(f"   data/processed/control_variables.csv")
    print(f"   data/final/analysis_dataset.csv")

    print(f"\nFinal analysis dataset:")
    print(f"   Companies: {len(analysis_df)}")
    print(f"   Variables: {len(analysis_df.columns)}")

    print(f"\nKey metrics available:")
    print(f"   ✅ Sharpe Ratio (DV for RQ1)")
    print(f"   ✅ Volatility (DV for RQ2)")
    print(f"   ✅ ESG Score (IV)")
    print(f"   ✅ E, S, G Pillars (IV for RQ3)")
    print(f"   ✅ Log Market Cap (control)")
    print(f"   ✅ Sector Dummies (control)")

    print(f"\nNext steps:")
    print(f"   python scripts/run_analysis.py")

    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
