"""Configuration constants for the benchmark project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Default cross-validation settings
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 1234

# Default hyperparameter settings for methods
DEFAULT_PARAMS = {
    "mean": {},
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    "catboost": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": None,
    },
    "hist_gradient_boosting": {
        "max_iter": 100,
        "max_depth": None,
        "learning_rate": 0.1,
    },
}

# Real dataset URLs
REAL_DATA_URLS = {
    "boston_housing": {
        "url": "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Boston.csv",
        "desc": "Housing values in Boston census tracts",
    },
    "california_housing": {
        "url": "https://raw.githubusercontent.com/szcf-weiya/ESL-CN/master/data/Housing/raw_data.csv",
        "desc": "California housing data",
    },
    "casp": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        "desc": "Physicochemical Properties of Protein Tertiary Structure",
    },
    "energy": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "desc": "Appliances Energy Prediction",
    },
    "air_quality": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/AirQualityUCI.csv",
        "desc": "Air Quality measurements",
    },
    "abalone": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
        "desc": "Abalone age prediction",
    },
}
