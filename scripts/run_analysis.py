"""
Master script to run statistical analysis.

This script:
1. Loads analysis dataset
2. Runs RQ1: ESG → Sharpe Ratio
3. Runs RQ2: ESG → Volatility
4. Runs RQ3: E, S, G Pillars → Performance & Risk
5. Runs diagnostic tests
6. Saves results

Usage:
    python scripts/run_analysis.py
"""
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.analysis.regression_models import run_rq1_sharpe_esg, run_rq2_volatility_esg, run_rq3_pillars
from src.analysis.diagnostics import run_diagnostics


def main():
    """
    Main function to orchestrate statistical analysis.
    """
    print("\n" + "=" * 60)
    print("ESG STOCK PERFORMANCE ANALYSIS - STATISTICAL ANALYSIS")
    print("=" * 60)

    # Load analysis dataset
    print("\n### Loading Analysis Dataset ###")
    analysis_file = "data/final/analysis_dataset.csv"

    try:
        df = pd.read_csv(analysis_file)
        print(f"✅ Loaded {len(df)} companies from {analysis_file}")
        print(f"\nColumns: {len(df.columns)}")
        print(f"Sample: {df.columns.tolist()[:10]}...")
    except FileNotFoundError:
        print(f"❌ File not found: {analysis_file}")
        print("Please run feature engineering first: python scripts/run_feature_engineering.py")
        return False

    # Create output directory
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    all_results = {}

    # RQ1: ESG → Sharpe Ratio
    print("\n\n" + "=" * 80)
    print("RESEARCH QUESTION 1")
    print("=" * 80)
    try:
        model_rq1, results_rq1 = run_rq1_sharpe_esg(df)
        all_results['rq1'] = {
            'model': model_rq1,
            'results': results_rq1
        }

        # Save results
        with open(output_dir / "rq1_results.txt", "w") as f:
            f.write(str(model_rq1.summary()))

        print(f"\n💾 Results saved to: {output_dir / 'rq1_results.txt'}")

    except Exception as e:
        print(f"\n❌ Error in RQ1: {e}")
        import traceback
        traceback.print_exc()

    # RQ2: ESG → Volatility
    print("\n\n" + "=" * 80)
    print("RESEARCH QUESTION 2")
    print("=" * 80)
    try:
        model_rq2, results_rq2 = run_rq2_volatility_esg(df)
        all_results['rq2'] = {
            'model': model_rq2,
            'results': results_rq2
        }

        # Save results
        with open(output_dir / "rq2_results.txt", "w") as f:
            f.write(str(model_rq2.summary()))

        print(f"\n💾 Results saved to: {output_dir / 'rq2_results.txt'}")

    except Exception as e:
        print(f"\n❌ Error in RQ2: {e}")
        import traceback
        traceback.print_exc()

    # RQ3: Pillars → Performance & Risk
    print("\n\n" + "=" * 80)
    print("RESEARCH QUESTION 3")
    print("=" * 80)
    try:
        models_rq3, results_rq3 = run_rq3_pillars(df)
        all_results['rq3'] = {
            'models': models_rq3,
            'results': results_rq3
        }

        # Save results
        if 'sharpe_pillars' in models_rq3:
            with open(output_dir / "rq3_sharpe_results.txt", "w") as f:
                f.write(str(models_rq3['sharpe_pillars'].summary()))

        if 'volatility_pillars' in models_rq3:
            with open(output_dir / "rq3_volatility_results.txt", "w") as f:
                f.write(str(models_rq3['volatility_pillars'].summary()))

        print(f"\n💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Error in RQ3: {e}")
        import traceback
        traceback.print_exc()

    # Run diagnostics
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC TESTS")
    print("=" * 80)

    diagnostics_results = {}

    if 'rq1' in all_results:
        try:
            model = all_results['rq1']['model']
            X = model.model.exog
            X_df = pd.DataFrame(X, columns=model.model.exog_names)

            diag_rq1 = run_diagnostics(model, X_df, "RQ1: Sharpe ~ ESG")
            diagnostics_results['rq1'] = diag_rq1

        except Exception as e:
            print(f"\n⚠️  Error running diagnostics for RQ1: {e}")

    if 'rq2' in all_results:
        try:
            model = all_results['rq2']['model']
            X = model.model.exog
            X_df = pd.DataFrame(X, columns=model.model.exog_names)

            diag_rq2 = run_diagnostics(model, X_df, "RQ2: Volatility ~ ESG")
            diagnostics_results['rq2'] = diag_rq2

        except Exception as e:
            print(f"\n⚠️  Error running diagnostics for RQ2: {e}")

    # Summary of all findings
    print("\n\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    if 'rq1' in all_results:
        r = all_results['rq1']['results']
        print(f"\n📊 RQ1: ESG Score → Sharpe Ratio")
        print(f"   Coefficient: {r['esg_coef']:.6f}")
        print(f"   P-value: {r['esg_pvalue']:.4f}")
        print(f"   Significant: {'✅ YES' if r['esg_significant'] else '❌ NO'}")
        print(f"   R²: {r['r_squared']:.4f}")
        print(f"   Interpretation: {'Higher ESG → Higher risk-adjusted returns' if r['esg_coef'] > 0 else 'Higher ESG → Lower risk-adjusted returns'}")

    if 'rq2' in all_results:
        r = all_results['rq2']['results']
        print(f"\n📊 RQ2: ESG Score → Volatility")
        print(f"   Coefficient: {r['esg_coef']:.6f}")
        print(f"   P-value: {r['esg_pvalue']:.4f}")
        print(f"   Significant: {'✅ YES' if r['esg_significant'] else '❌ NO'}")
        print(f"   R²: {r['r_squared']:.4f}")
        print(f"   Interpretation: {'Higher ESG → Lower volatility (more stable)' if r['esg_coef'] < 0 else 'Higher ESG → Higher volatility (less stable)'}")

    if 'rq3' in all_results:
        r = all_results['rq3']['results']
        if 'sharpe' in r:
            print(f"\n📊 RQ3a: ESG Pillars → Sharpe Ratio")
            print(f"   E coefficient: {r['sharpe']['e_coef']:.6f} (p={r['sharpe']['e_pvalue']:.4f})")
            print(f"   S coefficient: {r['sharpe']['s_coef']:.6f} (p={r['sharpe']['s_pvalue']:.4f})")
            print(f"   G coefficient: {r['sharpe']['g_coef']:.6f} (p={r['sharpe']['g_pvalue']:.4f})")
            print(f"   Dominant pillar: {r['sharpe']['dominant_pillar']}")

        if 'volatility' in r:
            print(f"\n📊 RQ3b: ESG Pillars → Volatility")
            print(f"   E coefficient: {r['volatility']['e_coef']:.6f} (p={r['volatility']['e_pvalue']:.4f})")
            print(f"   S coefficient: {r['volatility']['s_coef']:.6f} (p={r['volatility']['s_pvalue']:.4f})")
            print(f"   G coefficient: {r['volatility']['g_coef']:.6f} (p={r['volatility']['g_pvalue']:.4f})")
            print(f"   Dominant pillar: {r['volatility']['dominant_pillar']}")

    # Save summary
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, "w") as f:
        f.write("ESG STOCK PERFORMANCE ANALYSIS - SUMMARY OF FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        if 'rq1' in all_results:
            r = all_results['rq1']['results']
            f.write("RQ1: Do companies with higher ESG scores earn higher risk-adjusted returns?\n")
            f.write(f"  ESG Coefficient: {r['esg_coef']:.6f}\n")
            f.write(f"  P-value: {r['esg_pvalue']:.4f}\n")
            f.write(f"  Significant at 5%: {'YES' if r['esg_significant'] else 'NO'}\n")
            f.write(f"  R-squared: {r['r_squared']:.4f}\n\n")

        if 'rq2' in all_results:
            r = all_results['rq2']['results']
            f.write("RQ2: Do higher ESG scores reduce stock return volatility?\n")
            f.write(f"  ESG Coefficient: {r['esg_coef']:.6f}\n")
            f.write(f"  P-value: {r['esg_pvalue']:.4f}\n")
            f.write(f"  Significant at 5%: {'YES' if r['esg_significant'] else 'NO'}\n")
            f.write(f"  R-squared: {r['r_squared']:.4f}\n\n")

        if 'rq3' in all_results and 'sharpe' in all_results['rq3']['results']:
            r = all_results['rq3']['results']
            f.write("RQ3: Which ESG pillar drives risk-adjusted returns and volatility?\n")
            f.write(f"  Sharpe Ratio - Dominant: {r['sharpe']['dominant_pillar']}\n")
            f.write(f"  Volatility - Dominant: {r['volatility']['dominant_pillar']}\n")

    print(f"\n💾 Summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
