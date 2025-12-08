"""
Create visualizations for ESG analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def plot_esg_distribution(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Plot distribution of ESG scores."""
    print("\n[INFO] Creating ESG score distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['totalEsg'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('ESG Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of ESG Scores')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_esg = df['totalEsg'].mean()
    ax.axvline(mean_esg, color='red', linestyle='--', label=f'Mean: {mean_esg:.2f}')
    ax.legend()

    plt.tight_layout()
    output_path = Path(output_dir) / "esg_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to: {output_path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Plot correlation heatmap of key variables."""
    print("\n[INFO] Creating correlation heatmap...")

    # Select key columns
    key_cols = []
    for col in df.columns:
        if any(k in col.lower() for k in ['esg', 'sharpe', 'volatility', 'beta', 'return', 'market_cap']):
            if 'sector' not in col.lower():
                key_cols.append(col)

    if len(key_cols) < 2:
        print("[WARNING]  Not enough variables for correlation matrix")
        return

    corr_df = df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix of Key Variables')

    plt.tight_layout()
    output_path = Path(output_dir) / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to: {output_path}")
    plt.close()


def plot_esg_vs_sharpe(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Scatter plot of ESG score vs Sharpe ratio."""
    print("\n[INFO] Creating ESG vs Sharpe ratio scatter plot...")

    plot_df = df[['totalEsg', 'Sharpe_Ratio']].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(plot_df['totalEsg'], plot_df['Sharpe_Ratio'], alpha=0.5, s=50)

    # Add regression line
    z = np.polyfit(plot_df['totalEsg'], plot_df['Sharpe_Ratio'], 1)
    p = np.poly1d(z)
    ax.plot(plot_df['totalEsg'], p(plot_df['totalEsg']), "r--", alpha=0.8, label='Best fit line')

    ax.set_xlabel('ESG Score')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('ESG Score vs. Risk-Adjusted Returns (Sharpe Ratio)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = Path(output_dir) / "esg_vs_sharpe.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to: {output_path}")
    plt.close()


def plot_esg_vs_volatility(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Scatter plot of ESG score vs volatility."""
    print("\n[INFO] Creating ESG vs Volatility scatter plot...")

    plot_df = df[['totalEsg', 'Volatility']].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(plot_df['totalEsg'], plot_df['Volatility'], alpha=0.5, s=50, color='green')

    # Add regression line
    z = np.polyfit(plot_df['totalEsg'], plot_df['Volatility'], 1)
    p = np.poly1d(z)
    ax.plot(plot_df['totalEsg'], p(plot_df['totalEsg']), "r--", alpha=0.8, label='Best fit line')

    ax.set_xlabel('ESG Score')
    ax.set_ylabel('Volatility (Annualized)')
    ax.set_title('ESG Score vs. Stock Return Volatility')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.tight_layout()
    output_path = Path(output_dir) / "esg_vs_volatility.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to: {output_path}")
    plt.close()


def plot_sector_esg(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Box plot of ESG scores by sector."""
    print("\n[INFO] Creating ESG scores by sector plot...")

    # Reconstruct Sector from dummy variables
    sector_cols = [col for col in df.columns if col.startswith('Sector_')]

    # Create Sector column from dummies
    df_copy = df.copy()
    df_copy['Sector'] = 'Basic Materials'  # Baseline (dropped category)
    for col in sector_cols:
        sector_name = col.replace('Sector_', '')
        df_copy.loc[df_copy[col] == 1, 'Sector'] = sector_name

    plot_df = df_copy[['totalEsg', 'Sector']].dropna()

    # Get sector counts and filter sectors with < 5 companies
    sector_counts = plot_df['Sector'].value_counts()
    valid_sectors = sector_counts[sector_counts >= 5].index
    plot_df = plot_df[plot_df['Sector'].isin(valid_sectors)]

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df.boxplot(column='totalEsg', by='Sector', ax=ax, rot=45)
    ax.set_xlabel('Sector')
    ax.set_ylabel('ESG Score')
    ax.set_title('ESG Scores by Sector')
    plt.suptitle('')  # Remove default title

    plt.tight_layout()
    output_path = Path(output_dir) / "esg_by_sector.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to: {output_path}")
    plt.close()


def plot_pillar_comparison(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Bar plot comparing E, S, G pillar averages."""
    print("\n[INFO] Creating ESG pillars comparison plot...")

    pillars = {
        'Environmental': df['environmentScore'].mean(),
        'Social': df['socialScore'].mean(),
        'Governance': df['governanceScore'].mean()
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(pillars.keys(), pillars.values(), color=['green', 'blue', 'orange'], alpha=0.7)
    ax.set_ylabel('Average Score')
    ax.set_title('Average ESG Pillar Scores')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()
    output_path = Path(output_dir) / "pillar_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to: {output_path}")
    plt.close()


def create_all_plots(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Create all visualization plots."""
    print("=" * 60)
    print("Creating All Visualizations")
    print("=" * 60)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_esg_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_esg_vs_sharpe(df, output_dir)
    plot_esg_vs_volatility(df, output_dir)
    plot_sector_esg(df, output_dir)
    plot_pillar_comparison(df, output_dir)

    print("\n[OK] All visualizations created!")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/generate_report.py")
