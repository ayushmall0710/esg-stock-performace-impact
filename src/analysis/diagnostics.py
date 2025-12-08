"""
Diagnostic tests for regression models.
"""

from typing import Dict

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_diagnostics(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    model_name: str = "Model",
) -> Dict:
    """
    Run diagnostic tests on fitted OLS model.

    Tests:
    - Heteroskedasticity (Breusch-Pagan test)
    - Multicollinearity (VIF)
    - Normality of residuals (Jarque-Bera test)

    Args:
        model: Fitted statsmodels OLS model
        X: Design matrix (with constant)
        model_name: Name for reporting

    Returns:
        Dictionary of diagnostic results
    """
    print("\n" + "=" * 60)
    print(f"DIAGNOSTIC TESTS: {model_name}")
    print("=" * 60)

    diagnostics = {}

    # 1. Heteroskedasticity (Breusch-Pagan test)
    print("\n[CHECKING] 1. Heteroskedasticity Test (Breusch-Pagan)")
    print("\tH0: Homoskedasticity (constant variance)")
    print("\tH1: Heteroskedasticity (non-constant variance)")

    try:
        bp_test = het_breuschpagan(model.resid, X)
        lm_statistic, lm_pvalue, f_statistic, f_pvalue = bp_test

        diagnostics["bp_lm_stat"] = lm_statistic
        diagnostics["bp_pvalue"] = lm_pvalue
        diagnostics["heteroskedasticity"] = lm_pvalue < 0.05

        print(f"\tLM Statistic: {lm_statistic:.4f}")
        print(f"\tp-value: {lm_pvalue:.4f}")

        if lm_pvalue < 0.05:
            print("\t[WARNING] REJECT H0: Heteroskedasticity detected (p < 0.05)")
            print("\t→ Recommendation: Use robust standard errors (HC1 or HC3)")
        else:
            print("\t[OK] FAIL TO REJECT H0: No evidence of heteroskedasticity")

    except Exception as e:
        print(f"\t[ERROR] Error running test: {e}")
        diagnostics["heteroskedasticity"] = None

    # 2. Multicollinearity (VIF)
    print("\n[CHECKING] 2. Multicollinearity (Variance Inflation Factor)")
    print("\tRule of thumb: VIF > 10 indicates problematic multicollinearity")

    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]

        diagnostics["vif"] = vif_data

        print("\n   VIF values:")
        for idx, row in vif_data.iterrows():
            vif_val = row["VIF"]
            flag = "[WARNING]" if vif_val > 10 else "[OK]"
            print(f"\t{flag} {row['Variable']}: {vif_val:.2f}")

        high_vif = vif_data[vif_data["VIF"] > 10]
        if len(high_vif) > 0:
            print(f"\n   [WARNING] {len(high_vif)} variables with VIF > 10")
            print("\t→ Recommendation: Consider removing highly correlated variables")
        else:
            print("\n   [OK] No problematic multicollinearity detected")

        diagnostics["high_vif_count"] = len(high_vif)

    except Exception as e:
        print(f"\t[ERROR] Error calculating VIF: {e}")
        diagnostics["vif"] = None

    # 3. Normality of Residuals (Jarque-Bera test)
    print("\n[CHECKING] 3. Normality of Residuals (Jarque-Bera Test)")
    print("\tH0: Residuals are normally distributed")
    print("\tH1: Residuals are not normally distributed")

    try:
        jb_stat, jb_pvalue, skew, kurtosis = stats.jarque_bera(model.resid)

        diagnostics["jb_stat"] = jb_stat
        diagnostics["jb_pvalue"] = jb_pvalue
        diagnostics["residuals_normal"] = jb_pvalue >= 0.05

        print(f"\tJB Statistic: {jb_stat:.4f}")
        print(f"\tp-value: {jb_pvalue:.4f}")
        print(f"\tSkewness: {skew:.4f}")
        print(f"\tKurtosis: {kurtosis:.4f}")

        if jb_pvalue < 0.05:
            print("\t[WARNING] REJECT H0: Residuals are not normally distributed")
            print("\t→ Note: OLS is robust to non-normality with large samples")
        else:
            print("\t[OK] FAIL TO REJECT H0: Residuals appear normally distributed")

    except Exception as e:
        print(f"\t[ERROR] Error running test: {e}")
        diagnostics["residuals_normal"] = None

    # 4. Additional diagnostics
    print("\n[INFO] Additional Diagnostics:")
    print(f"\tR-squared: {model.rsquared:.4f}")
    print(f"\tAdjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"\tF-statistic: {model.fvalue:.4f} (p={model.f_pvalue:.4e})")
    print(f"\tAIC: {model.aic:.2f}")
    print(f"\tBIC: {model.bic:.2f}")

    diagnostics["r_squared"] = model.rsquared
    diagnostics["adj_r_squared"] = model.rsquared_adj
    diagnostics["aic"] = model.aic
    diagnostics["bic"] = model.bic

    # Summary recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    if diagnostics.get("heteroskedasticity"):
        print("[WARNING] Use robust standard errors (HC1 or HC3) due to heteroskedasticity")
    else:
        print("[OK] Standard errors appear valid")

    if diagnostics.get("high_vif_count", 0) > 0:
        print("[WARNING] Consider variable selection to reduce multicollinearity")
    else:
        print("[OK] No multicollinearity issues")

    if not diagnostics.get("residuals_normal"):
        print("[INFO] Residuals non-normal, but OLS is robust with large samples (n > 30)")

    return diagnostics


def fit_robust_model(
    model: sm.regression.linear_model.RegressionResultsWrapper, cov_type: str = "HC3"
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Refit model with robust standard errors.

    Args:
        model: Original OLS model
        cov_type: Type of robust covariance ('HC1', 'HC2', 'HC3')

    Returns:
        Model with robust standard errors
    """
    print(f"\n[PROCESSING] Refitting model with {cov_type} robust standard errors...")

    # Refit with robust covariance
    robust_model = model.get_robustcov_results(cov_type=cov_type)

    print("[OK] Robust model fitted")
    print("\nRobust results:")
    print(robust_model.summary())

    return robust_model


if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Run: python scripts/run_analysis.py")
