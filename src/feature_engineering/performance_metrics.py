"""
Calculate performance metrics: Sharpe ratio, excess returns, cumulative returns.
"""

import numpy as np
import pandas as pd


def calculate_performance_metrics(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
    return_col: str = "Return",
    excess_return_col: str = "Excess_Return",
) -> pd.DataFrame:
    """
    Calculate performance metrics for each ticker over the full time period.

    Metrics calculated:
    - Annualized excess return (mean excess return * 252)
    - Sharpe ratio (annualized)
    - Cumulative return
    - Annualized return

    Args:
        df: DataFrame with daily returns and excess returns
        ticker_col: Name of ticker column
        return_col: Name of return column
        excess_return_col: Name of excess return column

    Returns:
        DataFrame with one row per ticker and performance metrics
    """
    print("=" * 60)
    print("Calculating Performance Metrics")
    print("=" * 60)

    print("\n[INFO] Input data:")
    print(f"\tTotal records: {len(df)}")
    print(f"\tTickers: {df[ticker_col].nunique()}")

    # Group by ticker and calculate metrics
    print("\n[PROCESSING] Calculating metrics for each ticker...")

    results = []

    for ticker in df[ticker_col].unique():
        ticker_data = df[df[ticker_col] == ticker].copy()

        # Skip if insufficient data
        if len(ticker_data) < 200:  # Require at least 200 trading days (~8 months)
            print(f"\t[WARNING] Skipping {ticker}: only {len(ticker_data)} trading days")
            continue

        metrics = {"Ticker": ticker}

        # Number of trading days
        metrics["Trading_Days"] = len(ticker_data)

        # 1. Mean daily excess return
        mean_excess_return = ticker_data[excess_return_col].mean()
        metrics["Mean_Daily_Excess_Return"] = mean_excess_return

        # 2. Annualized excess return
        # Assuming 252 trading days per year
        metrics["Annualized_Excess_Return"] = mean_excess_return * 252

        # 3. Sharpe Ratio (annualized)
        # Sharpe = (mean excess return / std of excess returns) * sqrt(252)
        std_excess_return = ticker_data[excess_return_col].std()
        if std_excess_return > 0:
            metrics["Sharpe_Ratio"] = (
                mean_excess_return / std_excess_return
            ) * np.sqrt(252)
        else:
            metrics["Sharpe_Ratio"] = 0.0

        # 4. Cumulative return
        # (1 + R1) * (1 + R2) * ... * (1 + Rn) - 1
        metrics["Cumulative_Return"] = (1 + ticker_data[return_col]).prod() - 1

        # 5. Annualized return (geometric mean)
        # (1 + cumulative_return) ^ (252 / n_days) - 1
        n_days = len(ticker_data)
        metrics["Annualized_Return"] = (1 + metrics["Cumulative_Return"]) ** (
            252 / n_days
        ) - 1

        # 6. Mean daily return
        metrics["Mean_Daily_Return"] = ticker_data[return_col].mean()

        results.append(metrics)

    # Create DataFrame
    performance_df = pd.DataFrame(results)

    print(f"\n[OK] Calculated metrics for {len(performance_df)} tickers")

    # Summary statistics
    print("\n[STATS] Performance Metrics Summary:")
    print("\nSharpe Ratio:")
    print(f"\tMean: {performance_df['Sharpe_Ratio'].mean():.4f}")
    print(f"\tMedian: {performance_df['Sharpe_Ratio'].median():.4f}")
    print(f"\tStd: {performance_df['Sharpe_Ratio'].std():.4f}")
    print(f"\tMin: {performance_df['Sharpe_Ratio'].min():.4f}")
    print(f"\tMax: {performance_df['Sharpe_Ratio'].max():.4f}")

    print("\nAnnualized Excess Return:")
    print(f"\tMean: {performance_df['Annualized_Excess_Return'].mean():.2%}")
    print(f"\tMedian: {performance_df['Annualized_Excess_Return'].median():.2%}")
    print(f"\tMin: {performance_df['Annualized_Excess_Return'].min():.2%}")
    print(f"\tMax: {performance_df['Annualized_Excess_Return'].max():.2%}")

    print("\nCumulative Return:")
    print(f"\tMean: {performance_df['Cumulative_Return'].mean():.2%}")
    print(f"\tMedian: {performance_df['Cumulative_Return'].median():.2%}")
    print(f"\tMin: {performance_df['Cumulative_Return'].min():.2%}")
    print(f"\tMax: {performance_df['Cumulative_Return'].max():.2%}")

    # Display sample
    print("\n[INFO] Sample results:")
    display_cols = [
        "Ticker",
        "Sharpe_Ratio",
        "Annualized_Excess_Return",
        "Cumulative_Return",
        "Trading_Days",
    ]
    print(performance_df[display_cols].head(10))

    return performance_df


if __name__ == "__main__":
    # This is meant to be imported, but can test with sample data
    print("This module is designed to be imported.")
    print("Run: python scripts/run_feature_engineering.py")
