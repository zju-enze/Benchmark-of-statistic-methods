"""Run benchmarks and save results for the webapp."""

import sys
sys.path.insert(0, 'src')

import argparse
from pathlib import Path

import pandas as pd
from benchmark.datasets.synthetic import SYNTHETIC_FUNCTIONS
from benchmark.methods import (
    MeanPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)
from benchmark.evaluation import (
    run_benchmark_synthetic,
    run_benchmark_real,
    upload_results,
    upload_all_results,
    setup_and_upload,
)
from benchmark.datasets.real import REAL_DATA_FUNCTIONS
from benchmark.config import RESULTS_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for tree-based regression methods."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to git repository after running benchmarks.",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="Path to git repository for uploading results. "
        "Defaults to project root if not provided.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="origin",
        help="Remote name to push to (default: origin).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to push to (default: main).",
    )
    parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="URL of remote repository (e.g., https://github.com/user/repo.git). "
        "Required only if repository is not initialized.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only run synthetic benchmarks.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Only run real data benchmarks.",
    )
    return parser.parse_args()


def main():
    """Run benchmarks and optionally upload results."""
    args = parse_args()

    # Determine repo path
    if args.repo_path:
        repo_path = Path(args.repo_path)
    else:
        repo_path = Path(__file__).parent

    # Define methods to benchmark
    methods = {
        "Mean": lambda: MeanPredictor(),
        "RandomForest": lambda: RandomForestPredictor(n_estimators=50),
        "XGBoost": lambda: XGBoostPredictor(n_estimators=50),
    }

    # Run synthetic benchmarks
    if not args.real_only:
        print("Running synthetic benchmarks...")
        results_synthetic = run_benchmark_synthetic(
            data_names=["sim_friedman", "sim_checkerboard", "sim_linear", "sim_max"],
            method_factories=methods,
            structures=["indep", "ar1"],
            ns=[100, 200, 500],
            ps=[20, 50],
            n_folds=5,
            n_jobs=-1,
        )
        results_synthetic.to_csv(RESULTS_DIR / "benchmark_synthetic.csv", index=False)
        print(f"Saved {len(results_synthetic)} synthetic results")

    # Run real data benchmarks
    if not args.synthetic_only:
        # Real data benchmarks (only loaders that are implemented)
        available_real = {
            name: loader for name, loader in REAL_DATA_FUNCTIONS.items()
            if name in ["boston_housing", "california_housing", "casp", "abalone"]
        }

        print("Running real data benchmarks...")
        results_real = run_benchmark_real(
            data_loaders=available_real,
            method_factories=methods,
            n_folds=5,
            n_jobs=-1,
        )
        results_real.to_csv(RESULTS_DIR / "benchmark_real.csv", index=False)
        print(f"Saved {len(results_real)} real data results")

    print("\nBenchmark completed!")
    print("Now run: streamlit run src/benchmark/webapp/app.py")

    # Upload results if requested
    if args.upload:
        print("\n" + "=" * 50)
        print("Uploading results to git repository...")
        print("=" * 50)

        try:
            results = upload_all_results(
                repo_path=repo_path,
                remote=args.remote,
                branch=args.branch,
            )

            all_success = True
            for filename, result in results.items():
                if result.get("success"):
                    print(f"  {filename}: {result['message']}")
                else:
                    print(f"  {filename}: FAILED - {result.get('error', 'Unknown error')}")
                    all_success = False

            if all_success:
                print("\nAll results uploaded successfully!")
            else:
                print("\nSome uploads failed. Please check the errors above.")

        except Exception as e:
            print(f"\nUpload failed: {e}")
            print("\nTo manually upload results:")
            print(f"  1. Ensure git is configured in {repo_path}")
            print(f"  2. Results are saved in: {RESULTS_DIR}")
            print(f"  3. Run: cd {repo_path} && git add results/*.csv && git commit -m 'Update results' && git push")


if __name__ == "__main__":
    main()
