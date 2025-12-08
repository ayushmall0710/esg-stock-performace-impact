"""
Generate visualizations and summary report.

Usage:
    python scripts/generate_report.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.visualization.plots import create_all_plots


def main():
    """Generate all visualizations and reports."""
    print("\n" + "=" * 60)
    print("ESG STOCK PERFORMANCE ANALYSIS - REPORT GENERATION")
    print("=" * 60)

    # Load analysis dataset
    print("\n### Loading Analysis Dataset ###")
    analysis_file = "data/final/analysis_dataset.csv"

    try:
        df = pd.read_csv(analysis_file)
        print(f"[OK] Loaded {len(df)} companies from {analysis_file}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {analysis_file}")
        print(
            "Please run feature engineering first: python scripts/run_feature_engineering.py"
        )
        return False

    # Create visualizations
    print("\n### Generating Visualizations ###")
    try:
        create_all_plots(df, output_dir="outputs/figures")
    except Exception as e:
        print(f"\n[ERROR] Error creating visualizations: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Create descriptive statistics table
    print("\n### Creating Descriptive Statistics Table ###")
    try:
        # Select key variables
        desc_cols = []
        for col in df.columns:
            if any(
                k in col.lower()
                for k in ["esg", "sharpe", "volatility", "beta", "return", "market_cap"]
            ):
                if "sector" not in col.lower():
                    desc_cols.append(col)

        desc_stats = df[desc_cols].describe()

        # Save to CSV
        output_file = Path("outputs/tables/descriptive_statistics.csv")
        desc_stats.to_csv(output_file)
        print(f"[OK] Saved to: {output_file}")

        # Also save nicely formatted version
        print("\nDescriptive Statistics:")
        print(desc_stats.round(4))

    except Exception as e:
        print(f"\n[WARNING] Error creating descriptive statistics: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] REPORT GENERATION COMPLETE")
    print("=" * 60)
    print("\nOutputs created:")
    print("\toutputs/figures/ - 6 visualization plots")
    print("\toutputs/tables/descriptive_statistics.csv")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
