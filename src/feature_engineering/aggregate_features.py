"""
Aggregate all features into a single firm-level dataset for regression analysis.
"""
import pandas as pd
from pathlib import Path


def aggregate_all_features(master_file: str = "data/final/master_dataset.csv",
                           performance_df: pd.DataFrame = None,
                           risk_df: pd.DataFrame = None,
                           controls_df: pd.DataFrame = None,
                           output_file: str = "data/final/analysis_dataset.csv") -> pd.DataFrame:
    """
    Aggregate all features into a firm-level dataset.

    This combines:
    - ESG scores (from master dataset)
    - Performance metrics (calculated)
    - Risk metrics (calculated)
    - Control variables (calculated)

    Args:
        master_file: Path to master dataset (for ESG scores)
        performance_df: DataFrame with performance metrics
        risk_df: DataFrame with risk metrics
        controls_df: DataFrame with control variables
        output_file: Path to save final analysis dataset

    Returns:
        DataFrame with one row per company and all features
    """
    print("=" * 60)
    print("Aggregating All Features")
    print("=" * 60)

    # Load master dataset to get ESG scores
    print(f"\n📂 Loading master dataset from: {master_file}")
    master_df = pd.read_csv(master_file)

    # Get one row per ticker with ESG scores
    print(f"\n🔧 Extracting ESG scores (one row per ticker)...")

    # Find ESG columns
    esg_columns = [col for col in master_df.columns if any(
        keyword in col.lower() for keyword in ['esg', 'environment', 'social', 'governance', 'score', 'rating']
    )]

    # Also include Ticker and Company name if available
    base_columns = ['Ticker']
    if 'Company_Name' in master_df.columns:
        base_columns.append('Company_Name')
    elif 'Company' in master_df.columns:
        base_columns.append('Company')

    esg_df = master_df[base_columns + esg_columns].drop_duplicates(subset=['Ticker']).copy()

    print(f"   ✅ ESG data: {len(esg_df)} companies")
    print(f"   ESG columns: {esg_columns}")

    # Start with ESG data
    analysis_df = esg_df.copy()

    # Merge performance metrics
    if performance_df is not None:
        print(f"\n🔧 Merging performance metrics...")
        print(f"   Performance data: {len(performance_df)} companies")

        analysis_df = analysis_df.merge(performance_df, on='Ticker', how='inner')
        print(f"   After merge: {len(analysis_df)} companies")
    else:
        print(f"\n⚠️  No performance metrics provided")

    # Merge risk metrics
    if risk_df is not None:
        print(f"\n🔧 Merging risk metrics...")
        print(f"   Risk data: {len(risk_df)} companies")

        analysis_df = analysis_df.merge(risk_df, on='Ticker', how='inner')
        print(f"   After merge: {len(analysis_df)} companies")
    else:
        print(f"\n⚠️  No risk metrics provided")

    # Merge control variables
    if controls_df is not None:
        print(f"\n🔧 Merging control variables...")
        print(f"   Controls data: {len(controls_df)} companies")

        analysis_df = analysis_df.merge(controls_df, on='Ticker', how='inner')
        print(f"   After merge: {len(analysis_df)} companies")
    else:
        print(f"\n⚠️  No control variables provided")

    # Final validation
    print(f"\n🔍 Final validation...")

    # Check for duplicates
    duplicates = analysis_df['Ticker'].duplicated().sum()
    if duplicates > 0:
        print(f"   ⚠️  Found {duplicates} duplicate tickers, removing...")
        analysis_df = analysis_df.drop_duplicates(subset=['Ticker'], keep='first')

    # Check for missing values
    print(f"\n📊 Missing value summary:")
    missing_counts = analysis_df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(missing_counts) > 0:
        for col, count in missing_counts.head(10).items():
            pct = (count / len(analysis_df)) * 100
            print(f"   {col}: {count} ({pct:.1f}%)")
    else:
        print(f"   ✅ No missing values!")

    # Summary
    print("\n" + "=" * 60)
    print("AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"Final dataset:")
    print(f"   Companies: {len(analysis_df)}")
    print(f"   Variables: {len(analysis_df.columns)}")
    print(f"   Completeness: {((analysis_df.notna().sum().sum()) / (len(analysis_df) * len(analysis_df.columns)) * 100):.2f}%")

    print(f"\nColumn categories:")

    # Count column types
    esg_cols = [col for col in analysis_df.columns if any(k in col.lower() for k in ['esg', 'environmental', 'social', 'governance'])]
    perf_cols = [col for col in analysis_df.columns if any(k in col.lower() for k in ['return', 'sharpe'])]
    risk_cols = [col for col in analysis_df.columns if any(k in col.lower() for k in ['volatility', 'beta', 'deviation', 'drawdown', 'var'])]
    control_cols = [col for col in analysis_df.columns if any(k in col.lower() for k in ['market_cap', 'sector'])]

    print(f"   ESG variables: {len(esg_cols)}")
    print(f"   Performance variables: {len(perf_cols)}")
    print(f"   Risk variables: {len(risk_cols)}")
    print(f"   Control variables: {len(control_cols)}")

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_file, index=False)
    print(f"\n✅ Analysis dataset saved to: {output_file}")

    # Display sample
    print("\nData preview:")
    print(analysis_df.head())

    print("\nAll columns:")
    for i, col in enumerate(analysis_df.columns, 1):
        print(f"   {i}. {col}")

    return analysis_df


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_feature_engineering.py")
