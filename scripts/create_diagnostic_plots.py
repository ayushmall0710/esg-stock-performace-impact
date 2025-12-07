"""
Create diagnostic visualizations for regression models.

Usage:
    python scripts/create_diagnostic_plots.py
"""
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

def create_diagnostic_plots(model, X, y, model_name, output_dir):
    """Create 4-panel diagnostic plot for regression model."""

    print(f"\n[INFO] Creating diagnostic plots for {model_name}...")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Regression Diagnostics: {model_name}', fontsize=14, fontweight='bold')

    # Get residuals and fitted values
    residuals = model.resid
    fitted = model.fittedvalues

    # 1. Residuals vs Fitted (check for heteroskedasticity)
    ax1 = axes[0, 0]
    ax1.scatter(fitted, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted\n(Check for heteroskedasticity)')
    ax1.grid(True, alpha=0.3)

    # Add lowess smooth line
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lowess_fit = lowess(residuals, fitted, frac=0.3)
    ax1.plot(lowess_fit[:, 0], lowess_fit[:, 1], 'b-', linewidth=2, label='LOWESS')
    ax1.legend()

    # 2. Q-Q Plot (check for normality of residuals)
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot\n(Check for normality)')
    ax2.grid(True, alpha=0.3)

    # 3. Scale-Location Plot (check for homoskedasticity)
    ax3 = axes[1, 0]
    standardized_resid = np.sqrt(np.abs((residuals - residuals.mean()) / residuals.std()))
    ax3.scatter(fitted, standardized_resid, alpha=0.5, s=20)
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location Plot\n(Check for equal variance)')
    ax3.grid(True, alpha=0.3)

    # Add lowess smooth line
    lowess_fit2 = lowess(standardized_resid, fitted, frac=0.3)
    ax3.plot(lowess_fit2[:, 0], lowess_fit2[:, 1], 'r-', linewidth=2, label='LOWESS')
    ax3.legend()

    # 4. Residuals Histogram (check for normality)
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Residual Distribution\n(Check for normality)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f'{model_name.lower().replace(" ", "_")}_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    plt.close()


def create_vif_plot(vif_data, model_name, output_dir):
    """Create VIF (multicollinearity) plot."""

    print(f"\n[INFO] Creating VIF plot for {model_name}...")

    # Filter out constant and very high VIF (infinite)
    vif_filtered = vif_data[vif_data['VIF'] < 100].copy()
    vif_filtered = vif_filtered[vif_filtered['Variable'] != 'const']

    # Sort by VIF
    vif_filtered = vif_filtered.sort_values('VIF', ascending=True)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_filtered['VIF']]

    ax.barh(vif_filtered['Variable'], vif_filtered['VIF'], color=colors, alpha=0.7)
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (threshold)')
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=1, label='VIF = 5 (caution)')

    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax.set_ylabel('Variable', fontsize=12)
    ax.set_title(f'Multicollinearity Check: {model_name}\nLower is better (VIF < 10)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f'{model_name.lower().replace(" ", "_")}_vif.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    plt.close()


def calculate_vif(X):
    """Calculate VIF for each variable."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data


def main():
    """Generate all diagnostic visualizations."""

    print("\n" + "=" * 60)
    print("REGRESSION DIAGNOSTICS VISUALIZATION")
    print("=" * 60)

    # Load data
    print("\n### Loading Data ###")
    df = pd.read_csv('data/final/analysis_dataset.csv')
    print(f"Loaded {len(df)} companies")

    # Create output directory
    output_dir = Path("outputs/figures/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare sector columns
    sector_cols = [col for col in df.columns if col.startswith('Sector_')]

    # ========== RQ1: Sharpe Ratio ~ ESG ==========
    print("\n" + "=" * 60)
    print("RQ1: ESG Score → Sharpe Ratio")
    print("=" * 60)

    # Prepare data
    X_vars = ['totalEsg', 'Log_Market_Cap'] + sector_cols
    model_df = df[['Sharpe_Ratio'] + X_vars].dropna()

    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df['Sharpe_Ratio']

    # Fit model
    model_rq1 = sm.OLS(y, X).fit()

    # Create plots
    create_diagnostic_plots(model_rq1, X, y, 'RQ1 Sharpe Ratio', output_dir)

    # VIF
    vif_rq1 = calculate_vif(X)
    create_vif_plot(vif_rq1, 'RQ1 Sharpe Ratio', output_dir)

    # ========== RQ2: Volatility ~ ESG ==========
    print("\n" + "=" * 60)
    print("RQ2: ESG Score → Volatility")
    print("=" * 60)

    # Prepare data
    model_df = df[['Volatility'] + X_vars].dropna()

    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df['Volatility']

    # Fit model
    model_rq2 = sm.OLS(y, X).fit()

    # Create plots
    create_diagnostic_plots(model_rq2, X, y, 'RQ2 Volatility', output_dir)

    # VIF
    vif_rq2 = calculate_vif(X)
    create_vif_plot(vif_rq2, 'RQ2 Volatility', output_dir)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC PLOTS COMPLETE")
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - rq1_sharpe_ratio_diagnostics.png (4-panel diagnostic)")
    print("  - rq1_sharpe_ratio_vif.png (multicollinearity)")
    print("  - rq2_volatility_diagnostics.png (4-panel diagnostic)")
    print("  - rq2_volatility_vif.png (multicollinearity)")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
