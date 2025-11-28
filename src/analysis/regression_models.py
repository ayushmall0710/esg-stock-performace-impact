"""
OLS regression models for research questions.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple


def run_rq1_sharpe_esg(df: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict]:
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

    # Find ESG score column
    esg_col = None
    for col in df.columns:
        if 'esg' in col.lower() and ('score' in col.lower() or 'rating' in col.lower()):
            if not any(x in col.lower() for x in ['environmental', 'social', 'governance', '_e', '_s', '_g']):
                esg_col = col
                break

    if esg_col is None:
        # Try to find total ESG column another way
        esg_candidates = [col for col in df.columns if 'esg' in col.lower()]
        if esg_candidates:
            esg_col = esg_candidates[0]
            print(f"⚠️  Using {esg_col} as ESG score column")

    # Find sector dummy columns
    sector_cols = [col for col in df.columns if col.startswith('Sector_')]

    # Prepare variables
    y = df['Sharpe_Ratio']
    X_vars = [esg_col, 'Log_Market_Cap'] + sector_cols

    # Drop rows with missing values
    model_df = df[['Sharpe_Ratio'] + X_vars].dropna()

    print(f"\n📊 Model specification:")
    print(f"   DV: Sharpe_Ratio")
    print(f"   IV: {esg_col}")
    print(f"   Controls: Log_Market_Cap + {len(sector_cols)} sector dummies")
    print(f"   Sample size: {len(model_df)} (dropped {len(df) - len(model_df)} due to missing values)")

    # Build model
    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df['Sharpe_Ratio']

    # Fit OLS
    model = sm.OLS(y, X).fit()

    print("\n" + "=" * 60)
    print(model.summary())
    print("=" * 60)

    # Extract key results
    results = {
        'model_name': 'RQ1_Sharpe_ESG',
        'n_obs': model.nobs,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'esg_coef': model.params[esg_col],
        'esg_pvalue': model.pvalues[esg_col],
        'esg_significant': model.pvalues[esg_col] < 0.05
    }

    print(f"\n🔑 Key Results:")
    print(f"   ESG coefficient: {results['esg_coef']:.6f}")
    print(f"   ESG p-value: {results['esg_pvalue']:.4f}")
    print(f"   Significant at 5%: {'✅ YES' if results['esg_significant'] else '❌ NO'}")
    print(f"   R-squared: {results['r_squared']:.4f}")

    return model, results


def run_rq2_volatility_esg(df: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict]:
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

    # Find ESG score column
    esg_col = None
    for col in df.columns:
        if 'esg' in col.lower() and ('score' in col.lower() or 'rating' in col.lower()):
            if not any(x in col.lower() for x in ['environmental', 'social', 'governance', '_e', '_s', '_g']):
                esg_col = col
                break

    if esg_col is None:
        # Try to find total ESG column another way
        esg_candidates = [col for col in df.columns if 'esg' in col.lower()]
        if esg_candidates:
            esg_col = esg_candidates[0]
            print(f"⚠️  Using {esg_col} as ESG score column")

    sector_cols = [col for col in df.columns if col.startswith('Sector_')]

    # Prepare variables
    y = df['Volatility']
    X_vars = [esg_col, 'Log_Market_Cap'] + sector_cols

    # Drop rows with missing values
    model_df = df[['Volatility'] + X_vars].dropna()

    print(f"\n📊 Model specification:")
    print(f"   DV: Volatility")
    print(f"   IV: {esg_col}")
    print(f"   Controls: Log_Market_Cap + {len(sector_cols)} sector dummies")
    print(f"   Sample size: {len(model_df)}")

    # Build model
    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df['Volatility']

    # Fit OLS
    model = sm.OLS(y, X).fit()

    print("\n" + "=" * 60)
    print(model.summary())
    print("=" * 60)

    # Extract key results
    results = {
        'model_name': 'RQ2_Volatility_ESG',
        'n_obs': model.nobs,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'esg_coef': model.params[esg_col],
        'esg_pvalue': model.pvalues[esg_col],
        'esg_significant': model.pvalues[esg_col] < 0.05
    }

    print(f"\n🔑 Key Results:")
    print(f"   ESG coefficient: {results['esg_coef']:.6f}")
    print(f"   ESG p-value: {results['esg_pvalue']:.4f}")
    print(f"   Significant at 5%: {'✅ YES' if results['esg_significant'] else '❌ NO'}")
    print(f"   Interpretation: {'Higher ESG → Lower volatility' if results['esg_coef'] < 0 else 'Higher ESG → Higher volatility'}")
    print(f"   R-squared: {results['r_squared']:.4f}")

    return model, results


def run_rq3_pillars(df: pd.DataFrame) -> Tuple[Dict[str, sm.regression.linear_model.RegressionResultsWrapper], Dict]:
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

    # Find pillar columns
    e_col = None
    s_col = None
    g_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'environment' in col_lower and ('score' in col_lower or 'rating' in col_lower):
            e_col = col
        elif col_lower in ['e_score', 'e', 'environmental', 'environment']:
            e_col = col
        elif 'social' in col_lower and ('score' in col_lower or 'rating' in col_lower):
            s_col = col
        elif col_lower in ['s_score', 's', 'social']:
            s_col = col
        elif 'governance' in col_lower and ('score' in col_lower or 'rating' in col_lower):
            g_col = col
        elif col_lower in ['g_score', 'g', 'governance']:
            g_col = col

    if not all([e_col, s_col, g_col]):
        print(f"\n❌ Could not find all pillar columns:")
        print(f"   E: {e_col}")
        print(f"   S: {s_col}")
        print(f"   G: {g_col}")
        return {}, {}

    print(f"\n📊 Using pillars:")
    print(f"   E: {e_col}")
    print(f"   S: {s_col}")
    print(f"   G: {g_col}")

    sector_cols = [col for col in df.columns if col.startswith('Sector_')]

    models = {}
    results = {}

    # Model 1: Sharpe Ratio
    print(f"\n--- Model 3a: Sharpe Ratio ~ E + S + G ---")

    X_vars = [e_col, s_col, g_col, 'Log_Market_Cap'] + sector_cols
    model_df = df[['Sharpe_Ratio'] + X_vars].dropna()

    print(f"   Sample size: {len(model_df)}")

    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df['Sharpe_Ratio']

    model_sharpe = sm.OLS(y, X).fit()
    models['sharpe_pillars'] = model_sharpe

    print("\n" + "=" * 60)
    print(model_sharpe.summary())
    print("=" * 60)

    results['sharpe'] = {
        'e_coef': model_sharpe.params[e_col],
        'e_pvalue': model_sharpe.pvalues[e_col],
        's_coef': model_sharpe.params[s_col],
        's_pvalue': model_sharpe.pvalues[s_col],
        'g_coef': model_sharpe.params[g_col],
        'g_pvalue': model_sharpe.pvalues[g_col]
    }

    # Determine dominant pillar
    abs_coefs = {
        'E': abs(results['sharpe']['e_coef']),
        'S': abs(results['sharpe']['s_coef']),
        'G': abs(results['sharpe']['g_coef'])
    }
    dominant = max(abs_coefs, key=abs_coefs.get)
    results['sharpe']['dominant_pillar'] = dominant

    print(f"\n🔑 Pillar Coefficients (Sharpe Ratio):")
    print(f"   E: {results['sharpe']['e_coef']:.6f} (p={results['sharpe']['e_pvalue']:.4f})")
    print(f"   S: {results['sharpe']['s_coef']:.6f} (p={results['sharpe']['s_pvalue']:.4f})")
    print(f"   G: {results['sharpe']['g_coef']:.6f} (p={results['sharpe']['g_pvalue']:.4f})")
    print(f"   Dominant: {dominant}")

    # Model 2: Volatility
    print(f"\n--- Model 3b: Volatility ~ E + S + G ---")

    model_df = df[['Volatility'] + X_vars].dropna()
    print(f"   Sample size: {len(model_df)}")

    X = model_df[X_vars]
    X = sm.add_constant(X)
    y = model_df['Volatility']

    model_vol = sm.OLS(y, X).fit()
    models['volatility_pillars'] = model_vol

    print("\n" + "=" * 60)
    print(model_vol.summary())
    print("=" * 60)

    results['volatility'] = {
        'e_coef': model_vol.params[e_col],
        'e_pvalue': model_vol.pvalues[e_col],
        's_coef': model_vol.params[s_col],
        's_pvalue': model_vol.pvalues[s_col],
        'g_coef': model_vol.params[g_col],
        'g_pvalue': model_vol.pvalues[g_col]
    }

    abs_coefs = {
        'E': abs(results['volatility']['e_coef']),
        'S': abs(results['volatility']['s_coef']),
        'G': abs(results['volatility']['g_coef'])
    }
    dominant = max(abs_coefs, key=abs_coefs.get)
    results['volatility']['dominant_pillar'] = dominant

    print(f"\n🔑 Pillar Coefficients (Volatility):")
    print(f"   E: {results['volatility']['e_coef']:.6f} (p={results['volatility']['e_pvalue']:.4f})")
    print(f"   S: {results['volatility']['s_coef']:.6f} (p={results['volatility']['s_pvalue']:.4f})")
    print(f"   G: {results['volatility']['g_coef']:.6f} (p={results['volatility']['g_pvalue']:.4f})")
    print(f"   Dominant: {dominant}")

    return models, results


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_analysis.py")
