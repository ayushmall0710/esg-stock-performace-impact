"""
OLS regression models for research questions.
"""

from typing import Dict, Tuple

import pandas as pd
import statsmodels.api as sm


def run_rq1_sharpe_esg(
    df: pd.DataFrame,
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict]:
    """
    RQ1: Do companies with higher ESG scores earn higher risk-adjusted returns?

    Model: Sharpe Ratio = β0 + β1*ESG_Score + β2*log_mcap + β3*sector_dummies + ε

    Args:
        df: Analysis dataset with all variables

    Returns:
        Tuple of (fitted model, summary dict)
    """
    print("=" * 60)
    print("RQ1: ESG Score → Sharpe Ratio")
    print("=" * 60)

    # Use known column names
    esg_col = "totalEsg"
    sector_cols = [col for col in df.columns if col.startswith("Sector_")]

    # Prepare variables
    y = df["Sharpe_Ratio"]
    X_vars = [esg_col, "Log_Market_Cap"] + sector_cols

    # Drop rows with missing values
    model_df = df[["Sharpe_Ratio"] + X_vars].dropna()

    print("\n[INFO] Model specification:")
    print("\tDV: Sharpe_Ratio")
    print(f"\tIV: {esg_col}")
    print(f"\tControls: Log_Market_Cap + {len(sector_cols)} sector dummies")
    print(
        f"\tSample size: {len(model_df)} (dropped {len(df) - len(model_df)} due to missing values)"
    )

    # Build model
    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df["Sharpe_Ratio"]

    # Fit OLS
    model = sm.OLS(y, X).fit()

    print("\n" + "=" * 60)
    print(model.summary())
    print("=" * 60)

    # Extract key results
    results = {
        "model_name": "RQ1_Sharpe_ESG",
        "n_obs": model.nobs,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "esg_coef": model.params[esg_col],
        "esg_pvalue": model.pvalues[esg_col],
        "esg_significant": model.pvalues[esg_col] < 0.05,
    }

    print("\n[KEY] Key Results:")
    print(f"\tESG coefficient: {results['esg_coef']:.6f}")
    print(f"\tESG p-value: {results['esg_pvalue']:.4f}")
    print(f"\tSignificant at 5%: {'[YES]' if results['esg_significant'] else '[NO]'}")
    print(f"\tR-squared: {results['r_squared']:.4f}")

    return model, results


def run_rq2_volatility_esg(
    df: pd.DataFrame,
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict]:
    """
    RQ2: Do higher ESG scores reduce stock return volatility?

    Model: Volatility = β0 + β1*ESG_Score + β2*log_mcap + β3*sector_dummies + ε

    Args:
        df: Analysis dataset

    Returns:
        Tuple of (fitted model, summary dict)
    """
    print("\n\n" + "=" * 60)
    print("RQ2: ESG Score → Volatility")
    print("=" * 60)

    # Use known column names
    esg_col = "totalEsg"
    sector_cols = [col for col in df.columns if col.startswith("Sector_")]

    # Prepare variables
    y = df["Volatility"]
    X_vars = [esg_col, "Log_Market_Cap"] + sector_cols

    # Drop rows with missing values
    model_df = df[["Volatility"] + X_vars].dropna()

    print("\n[INFO] Model specification:")
    print("\tDV: Volatility")
    print(f"\tIV: {esg_col}")
    print(f"\tControls: Log_Market_Cap + {len(sector_cols)} sector dummies")
    print(f"\tSample size: {len(model_df)}")

    # Build model
    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df["Volatility"]

    # Fit OLS
    model = sm.OLS(y, X).fit()

    print("\n" + "=" * 60)
    print(model.summary())
    print("=" * 60)

    # Extract key results
    results = {
        "model_name": "RQ2_Volatility_ESG",
        "n_obs": model.nobs,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "esg_coef": model.params[esg_col],
        "esg_pvalue": model.pvalues[esg_col],
        "esg_significant": model.pvalues[esg_col] < 0.05,
    }

    print("\n[KEY] Key Results:")
    print(f"\tESG coefficient: {results['esg_coef']:.6f}")
    print(f"\tESG p-value: {results['esg_pvalue']:.4f}")
    print(f"\tSignificant at 5%: {'[YES]' if results['esg_significant'] else '[NO]'}")
    print(
        f"\tInterpretation: {'Higher ESG → Lower volatility' if results['esg_coef'] < 0 else 'Higher ESG → Higher volatility'}"
    )
    print(f"\tR-squared: {results['r_squared']:.4f}")

    return model, results


def run_rq3_pillars(
    df: pd.DataFrame,
) -> Tuple[Dict[str, sm.regression.linear_model.RegressionResultsWrapper], Dict]:
    """
    RQ3: Which ESG pillar (E, S, G) drives risk-adjusted returns and volatility?

    Models:
    - Sharpe Ratio = β0 + β1*E + β2*S + β3*G + β4*log_mcap + β5*sector_dummies + ε
    - Volatility = β0 + β1*E + β2*S + β3*G + β4*log_mcap + β5*sector_dummies + ε

    Args:
        df: Analysis dataset

    Returns:
        Tuple of (dict of fitted models, summary dict)
    """
    print("\n\n" + "=" * 60)
    print("RQ3: ESG Pillars (E, S, G) → Performance & Risk")
    print("=" * 60)

    # Use known pillar column names
    e_col = "environmentScore"
    s_col = "socialScore"
    g_col = "governanceScore"

    print("\n[INFO] Using pillars:")
    print(f"\tE: {e_col}")
    print(f"\tS: {s_col}")
    print(f"\tG: {g_col}")

    sector_cols = [col for col in df.columns if col.startswith("Sector_")]

    models = {}
    results = {}

    # Model 1: Sharpe Ratio
    print("\n--- Model 3a: Sharpe Ratio ~ E + S + G ---")

    X_vars = [e_col, s_col, g_col, "Log_Market_Cap"] + sector_cols
    model_df = df[["Sharpe_Ratio"] + X_vars].dropna()

    print(f"\tSample size: {len(model_df)}")

    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df["Sharpe_Ratio"]

    model_sharpe = sm.OLS(y, X).fit()
    models["sharpe_pillars"] = model_sharpe

    print("\n" + "=" * 60)
    print(model_sharpe.summary())
    print("=" * 60)

    results["sharpe"] = {
        "e_coef": model_sharpe.params[e_col],
        "e_pvalue": model_sharpe.pvalues[e_col],
        "s_coef": model_sharpe.params[s_col],
        "s_pvalue": model_sharpe.pvalues[s_col],
        "g_coef": model_sharpe.params[g_col],
        "g_pvalue": model_sharpe.pvalues[g_col],
    }

    # Determine dominant pillar
    abs_coefs = {
        "E": abs(results["sharpe"]["e_coef"]),
        "S": abs(results["sharpe"]["s_coef"]),
        "G": abs(results["sharpe"]["g_coef"]),
    }
    dominant = max(abs_coefs, key=abs_coefs.get)
    results["sharpe"]["dominant_pillar"] = dominant

    print("\n[KEY] Pillar Coefficients (Sharpe Ratio):")
    print(
        f"\tE: {results['sharpe']['e_coef']:.6f} (p={results['sharpe']['e_pvalue']:.4f})"
    )
    print(
        f"\tS: {results['sharpe']['s_coef']:.6f} (p={results['sharpe']['s_pvalue']:.4f})"
    )
    print(
        f"\tG: {results['sharpe']['g_coef']:.6f} (p={results['sharpe']['g_pvalue']:.4f})"
    )
    print(f"\tDominant: {dominant}")

    # Model 2: Volatility
    print("\n--- Model 3b: Volatility ~ E + S + G ---")

    model_df = df[["Volatility"] + X_vars].dropna()
    print(f"\tSample size: {len(model_df)}")

    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df["Volatility"]

    model_vol = sm.OLS(y, X).fit()
    models["volatility_pillars"] = model_vol

    print("\n" + "=" * 60)
    print(model_vol.summary())
    print("=" * 60)

    results["volatility"] = {
        "e_coef": model_vol.params[e_col],
        "e_pvalue": model_vol.pvalues[e_col],
        "s_coef": model_vol.params[s_col],
        "s_pvalue": model_vol.pvalues[s_col],
        "g_coef": model_vol.params[g_col],
        "g_pvalue": model_vol.pvalues[g_col],
    }

    abs_coefs = {
        "E": abs(results["volatility"]["e_coef"]),
        "S": abs(results["volatility"]["s_coef"]),
        "G": abs(results["volatility"]["g_coef"]),
    }
    dominant = max(abs_coefs, key=abs_coefs.get)
    results["volatility"]["dominant_pillar"] = dominant

    print("\n[KEY] Pillar Coefficients (Volatility):")
    print(
        f"\tE: {results['volatility']['e_coef']:.6f} (p={results['volatility']['e_pvalue']:.4f})"
    )
    print(
        f"\tS: {results['volatility']['s_coef']:.6f} (p={results['volatility']['s_pvalue']:.4f})"
    )
    print(
        f"\tG: {results['volatility']['g_coef']:.6f} (p={results['volatility']['g_pvalue']:.4f})"
    )
    print(f"\tDominant: {dominant}")

    return models, results


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_analysis.py")
