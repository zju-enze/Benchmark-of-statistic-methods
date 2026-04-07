# Benchmarking Tree Regressions (Python)

A Python implementation for benchmarking tree-based regression methods, inspired by the [R version](https://github.com/szcf-weiya/benchmark.tree.regressions).

## Overview

This project benchmarks various tree-based regression methods on both synthetic and real datasets, providing:

- **5 synthetic data generators** with different covariance structures
- **14 real-world datasets** with auto-download support
- **Multiple ML methods** (XGBoost, CatBoost, RandomForest, etc.)
- **5-fold cross-validation** with MSE and runtime metrics
- **Interactive visualization** via Streamlit

## Installation

```bash
pip install -e .
```

## Quick Start

### Run Benchmarks (Command Line)

The easiest way to run benchmarks:

```bash
python run_benchmark.py
```

Optional flags:

```bash
# Run only synthetic benchmarks
python run_benchmark.py --synthetic-only

# Run only real data benchmarks
python run_benchmark.py --real-only

# Run and upload results to git repository
python run_benchmark.py --upload

# Specify custom repository path and remote
python run_benchmark.py --upload --repo-path /path/to/repo --remote origin --branch main
```

### Run Benchmarks (Python API)

```python
from benchmark.datasets.synthetic import SYNTHETIC_FUNCTIONS
from benchmark.methods import XGBoostPredictor, RandomForestPredictor, MeanPredictor
from benchmark.evaluation import run_benchmark_synthetic

# Define methods
methods = {
    "XGBoost": lambda: XGBoostPredictor(n_estimators=100, max_depth=6),
    "RandomForest": lambda: RandomForestPredictor(n_estimators=500),
    "Mean": lambda: MeanPredictor(),
}

# Run synthetic benchmarks
results = run_benchmark_synthetic(
    data_names=["sim_friedman", "sim_linear"],
    method_factories=methods,
    structures=["indep", "ar1"],
    ns=[100, 200],
    ps=[20, 50],
)

# Save results
results.to_csv("results/benchmark_synthetic.csv", index=False)
```

### Upload Results to Git

Share your benchmark results by uploading to a git repository:

```bash
# Upload after running benchmarks
python run_benchmark.py --upload

# With custom repository
python run_benchmark.py --upload --repo-path /path/to/repo --remote-url https://github.com/user/repo.git
```

The upload function will:
1. Stage the results CSV files (`benchmark_synthetic.csv`, `benchmark_real.csv`)
2. Create a commit with system info (OS, Python version, timestamp)
3. Push to the specified remote and branch

### Run Streamlit App

```bash
streamlit run src/benchmark/webapp/app.py
```

## Project Structure

```
benchmark-tree-regressions-py/
├── .github/
│   ├── workflows/
│   │   └── add_dataset.yml       # GitHub Action for adding datasets
│   └── scripts/
│       └── add_dataset.py       # Script to add new dataset
├── pyproject.toml
├── README.md
├── src/benchmark/
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── datasets/
│   │   ├── synthetic.py           # Synthetic data generators
│   │   └── real.py               # Real data loaders
│   ├── methods/
│   │   ├── base.py               # BasePredictor class
│   │   ├── xgboost.py            # XGBoost wrapper
│   │   ├── catboost.py           # CatBoost wrapper
│   │   ├── random_forest.py      # sklearn RF wrapper
│   │   ├── bart.py               # BART approximation
│   │   ├── mars.py               # MARS wrapper
│   │   └── xbart.py              # XBART wrapper
│   ├── evaluation/
│   │   ├── cross_validation.py   # K-fold CV
│   │   ├── evaluate.py           # Single evaluation
│   │   ├── benchmark.py          # Full benchmark runner
│   │   └── upload.py            # Git upload utilities
│   └── webapp/
│       └── app.py                # Streamlit app
├── data/                          # Downloaded data
└── results/                       # Saved results
```

## Methods

| Method | Class | Package |
|--------|-------|---------|
| XGBoost | `XGBoostPredictor` | xgboost |
| CatBoost | `CatBoostPredictor` | catboost |
| Random Forest | `RandomForestPredictor` | scikit-learn |
| BART (approx) | `BARTPredictor` | sklearn |
| MARS | `MARSPredictor` | py-earth |
| XBART | `XBARTPredictor` | pyxbart |
| Mean | `MeanPredictor` | - |

## Synthetic Data

### Data Models

- **Friedman**: $f(x) = 10\sin(\pi x_1x_2) + 20(x_3-0.5)^2 + 10x_4 + 5x_5$
- **Checkerboard**: $f(x) = 2x_1x_2 + 2x_3x_4$
- **Linear**: $f(x) = 2x_1 + 2x_2 + 4x_3$
- **Max**: $f(x) = \max(x_1, x_2, x_3)$
- **Single Index**: $f(x) = 10\sqrt{a} + \sin(5a)$

### Covariance Structures

- **Independent**: $X_{ij} \sim N(0, 1)$
- **AR(1)**: $\Sigma_{jk} = \rho^{|j-k|}$
- **AR(1)+**: $\Sigma_{jk} = \rho_1^{|j-k|} + \rho_2 I(j \neq k)$
- **Factor**: $\mathbf{X = (BF)^\top + \epsilon}$

## Real Datasets

The following real datasets are supported:

- Boston Housing
- California Housing
- CASP (Protein Structure)
- Energy (Appliances)
- Air Quality
- Abalone
- Wine Quality
- And more...

## Contributing New Datasets via GitHub Action

This project supports an automated GitHub Action bot that allows users to contribute new datasets for benchmarking by simply opening an issue.

### How It Works

Open a new GitHub issue with a title in the following format:

```
@new-csv DataName DataURL Idx_of_Y
```

Where:
- **DataName**: A short name for the dataset (will be used as identifier)
- **DataURL**: Full URL to a CSV file (publicly accessible)
- **Idx_of_Y**: 1-indexed column number of the response/target variable

### Example

To add the abalone dataset (response is the 9th column):

```
@new-csv abalone https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/abalone.csv 9
```

The bot will automatically:
1. Download the CSV file
2. Add it to the project with a loader function
3. Create a commit
4. Open a pull request

### Requirements

- The CSV file must be publicly accessible via URL
- The file should not require authentication
- The response variable is assumed to be in the specified column (1-indexed)
- The CSV is assumed to have no header row

## License

MIT License
