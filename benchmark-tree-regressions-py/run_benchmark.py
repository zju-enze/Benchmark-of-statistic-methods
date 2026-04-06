"""Run benchmarks and save results for the webapp."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from benchmark.datasets.synthetic import SYNTHETIC_FUNCTIONS
from benchmark.methods import (
    MeanPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)
from benchmark.evaluation import run_benchmark_synthetic, run_benchmark_real
from benchmark.datasets.real import REAL_DATA_FUNCTIONS

# Define methods to benchmark
methods = {
    "Mean": lambda: MeanPredictor(),
    "RandomForest": lambda: RandomForestPredictor(n_estimators=50),
    "XGBoost": lambda: XGBoostPredictor(n_estimators=50),
}

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
results_synthetic.to_csv("results/benchmark_synthetic.csv", index=False)
print(f"Saved {len(results_synthetic)} synthetic results")

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
results_real.to_csv("results/benchmark_real.csv", index=False)
print(f"Saved {len(results_real)} real data results")

print("Done! Now run: streamlit run src/benchmark/webapp/app.py")
