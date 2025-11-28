"""
Calculate risk metrics: volatility, beta, downside deviation.
"""
import pandas as pd
import numpy as np
from scipy import stats


def calculate_risk_metrics(df: pd.DataFrame,
                           ticker_col: str = 'Ticker',
                           return_col: str = 'Return',
                           excess_return_col: str = 'Excess_Return',
                           market_return_col: str = 'Market_Return') -> pd.DataFrame:
    """
    Calculate risk metrics for each ticker over the full time period.

    Metrics calculated:
    - Annualized volatility (standard deviation of returns)
    - Beta (systematic risk relative to market)
    - Downside deviation (semi-deviation of negative returns)

    Args:
        df: DataFrame with daily returns
        ticker_col: Name of ticker column
        return_col: Name of return column
        excess_return_col: Name of excess return column
        market_return_col: Name of market return column

    Returns:
        DataFrame with one row per ticker and risk metrics
    """
    print("=" * 60)
    print("Calculating Risk Metrics")
    print("=" * 60)

    print(f"\n📊 Input data:")
    print(f"   Total records: {len(df)}")
    print(f"   Tickers: {df[ticker_col].nunique()}")

    # Check if market returns are available
    has_market_returns = market_return_col in df.columns and df[market_return_col].notna().any()

    if not has_market_returns:
        print(f"\n⚠️  Market returns not available, beta calculation will be skipped")

    # Group by ticker and calculate metrics
    print(f"\n🔧 Calculating metrics for each ticker...")

    results = []

    for ticker in df[ticker_col].unique():
        ticker_data = df[df[ticker_col] == ticker].copy()

        # Skip if insufficient data
        if len(ticker_data) < 200:
            continue

        metrics = {'Ticker': ticker}

        # 1. Volatility (annualized standard deviation of returns)
        daily_std = ticker_data[return_col].std()
        metrics['Volatility'] = daily_std * np.sqrt(252)

        # 2. Beta (if market returns available)
        if has_market_returns:
            # Remove NaN values
            valid_data = ticker_data[[return_col, market_return_col]].dropna()

            if len(valid_data) >= 100:  # Need sufficient data for regression
                # Calculate beta using covariance method
                # Beta = Cov(stock_return, market_return) / Var(market_return)
                cov_matrix = np.cov(valid_data[return_col], valid_data[market_return_col])
                covariance = cov_matrix[0, 1]
                market_variance = cov_matrix[1, 1]

                if market_variance > 0:
                    metrics['Beta'] = covariance / market_variance
                else:
                    metrics['Beta'] = np.nan

                # Alternative: Use linear regression for beta (should give same result)
                # slope, intercept, r_value, p_value, std_err = stats.linregress(
                #     valid_data[market_return_col], valid_data[return_col]
                # )
                # metrics['Beta'] = slope
                # metrics['Beta_R_Squared'] = r_value ** 2

            else:
                metrics['Beta'] = np.nan
        else:
            metrics['Beta'] = np.nan

        # 3. Downside deviation (semi-deviation - only negative returns)
        # This measures downside risk
        negative_returns = ticker_data[ticker_data[return_col] < 0][return_col]

        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            metrics['Downside_Deviation'] = downside_std * np.sqrt(252)
        else:
            metrics['Downside_Deviation'] = 0.0

        # 4. Standard deviation of excess returns (for Sharpe calculation verification)
        metrics['Excess_Return_Std'] = ticker_data[excess_return_col].std() * np.sqrt(252)

        # 5. Value at Risk (VaR) - 5% worst case daily return
        metrics['VaR_5pct'] = ticker_data[return_col].quantile(0.05)

        # 6. Maximum drawdown
        cumulative_returns = (1 + ticker_data[return_col]).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['Max_Drawdown'] = drawdown.min()

        results.append(metrics)

    # Create DataFrame
    risk_df = pd.DataFrame(results)

    print(f"\n✅ Calculated metrics for {len(risk_df)} tickers")

    # Summary statistics
    print("\n📈 Risk Metrics Summary:")

    print(f"\nVolatility (Annualized):")
    print(f"   Mean: {risk_df['Volatility'].mean():.2%}")
    print(f"   Median: {risk_df['Volatility'].median():.2%}")
    print(f"   Min: {risk_df['Volatility'].min():.2%}")
    print(f"   Max: {risk_df['Volatility'].max():.2%}")

    if has_market_returns and risk_df['Beta'].notna().any():
        print(f"\nBeta:")
        print(f"   Mean: {risk_df['Beta'].mean():.4f}")
        print(f"   Median: {risk_df['Beta'].median():.4f}")
        print(f"   Min: {risk_df['Beta'].min():.4f}")
        print(f"   Max: {risk_df['Beta'].max():.4f}")
        print(f"   % with Beta > 1 (more volatile than market): {(risk_df['Beta'] > 1).sum() / len(risk_df) * 100:.1f}%")

    print(f"\nDownside Deviation:")
    print(f"   Mean: {risk_df['Downside_Deviation'].mean():.2%}")
    print(f"   Median: {risk_df['Downside_Deviation'].median():.2%}")

    print(f"\nMax Drawdown:")
    print(f"   Mean: {risk_df['Max_Drawdown'].mean():.2%}")
    print(f"   Median: {risk_df['Max_Drawdown'].median():.2%}")
    print(f"   Worst: {risk_df['Max_Drawdown'].min():.2%}")

    # Display sample
    print("\n📊 Sample results:")
    if has_market_returns:
        display_cols = ['Ticker', 'Volatility', 'Beta', 'Downside_Deviation', 'Max_Drawdown']
    else:
        display_cols = ['Ticker', 'Volatility', 'Downside_Deviation', 'Max_Drawdown']
    print(risk_df[display_cols].head(10))

    return risk_df


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_feature_engineering.py")
