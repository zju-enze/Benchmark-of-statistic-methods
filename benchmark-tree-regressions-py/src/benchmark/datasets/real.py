"""Real data loaders with auto-download support."""

import hashlib
import os
import warnings
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

from ..config import DATA_DIR


def _download_with_retry(
    url: str,
    dest: Path,
    max_attempts: int = 5,
    wait_time: float = 5.0,
) -> None:
    """Download a file with retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            with open(dest, "wb") as f:
                f.write(response.content)
            return
        except requests.RequestException:
            if attempt < max_attempts - 1:
                import time
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts")


def _get_local_path(dataset_name: str, filename: str) -> Path:
    """Get local path for a dataset file, downloading if necessary."""
    # This would be populated with actual download URLs from config
    return DATA_DIR / dataset_name / filename


# Real dataset metadata - URLs and descriptions
REAL_DATA_INFO: Dict[str, Dict[str, str]] = {
    "BostonHousing": {
        "url": "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Boston.csv",
        "desc": "Housing values in Boston census tracts",
        "filename": "Boston.csv",
    },
    "CaliforniaHousing": {
        "url": "https://raw.githubusercontent.com/szcf-weiya/ESL-CN/master/data/Housing/raw_data.csv",
        "desc": "California housing data",
        "filename": "california_housing.csv",
    },
    "CASP": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        "desc": "Physicochemical Properties of Protein Tertiary Structure",
        "filename": "CASP.csv",
    },
    "Energy": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "desc": "Appliances Energy Prediction",
        "filename": "ENB2012_data.xlsx",
    },
    "AirQuality": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/AirQualityUCI.csv",
        "desc": "Air Quality measurements",
        "filename": "AirQualityUCI.csv",
    },
    "BiasCorrection": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00448/Bias_correction_ucl.csv",
        "desc": "Bias correction of numerical weather model",
        "filename": "Bias_correction_ucl.csv",
    },
    "ElectricalStability": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv",
        "desc": "Electrical Grid Stability Simulated Data",
        "filename": "Data_for_UCI_named.csv",
    },
    "GasTurbine": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00455/gt_2011.csv",
        "desc": "Gas Turbine CO and NOx Emission Data",
        "filename": "gt_2011.csv",
    },
    "abalone": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
        "desc": "Abalone age prediction",
        "filename": "abalone.csv",
    },
    "WineQualityRed": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv",
        "desc": "Red wine quality prediction",
        "filename": "winequality-red.csv",
    },
}


def _load_or_download(dataset_name: str) -> pd.DataFrame:
    """Load a dataset, downloading if necessary."""
    info = REAL_DATA_INFO.get(dataset_name)
    if info is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    filepath = DATA_DIR / dataset_name / info["filename"]

    # For multi-file datasets like GasTurbine, check for first file
    if not filepath.exists():
        _download_with_retry(info["url"], filepath)

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix == ".xlsx":
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def _to_design_matrix(df: pd.DataFrame, drop_cols: list = None) -> Tuple[NDArray, NDArray]:
    """Convert DataFrame to X matrix (with dummy encoding) and y vector."""
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Get y as last column, X as everything else
    y = df.iloc[:, -1].values
    X_df = df.iloc[:, :-1]

    # One-hot encode categorical columns
    X_df = pd.get_dummies(X_df, drop_first=False)

    return X_df.values, y


def load_boston() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Boston Housing dataset.
    Response: log(medv)
    """
    df = _load_or_download("BostonHousing")
    # Drop the 'Unnamed: 0' column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Response is 'medv', log transform
    y = np.log(df["medv"].values)
    X = pd.get_dummies(df.drop(columns=["medv"]), drop_first=False).values
    return X, y


def load_california_housing() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    California Housing dataset.
    Response: log(MedianHouseValue)
    """
    df = _load_or_download("CaliforniaHousing")
    # Response is first column
    y = np.log(df.iloc[:, 0].values)
    X = pd.get_dummies(df.drop(columns=[df.columns[0]]), drop_first=False).values
    return X, y


def load_casp() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    CASP protein tertiary structure dataset.
    Response: first column (RMSD)
    """
    df = _load_or_download("CASP")
    y = df.iloc[:, 0].values
    X = pd.get_dummies(df.iloc[:, 1:], drop_first=False).values
    return X, y # type: ignore


def load_energy() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Appliances Energy Prediction dataset.
    Response: Appliances energy consumption (T1 in original)
    """
    df = _load_or_download("Energy")
    # Drop date column, response is second column
    y = df.iloc[:, 1].values
    X = pd.get_dummies(df.drop(columns=[df.columns[0], df.columns[1]]), drop_first=False).values
    return X, y # type: ignore


def load_air_quality() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Air Quality dataset.
    Response: T (temperature)
    """
    df = _load_or_download("AirQuality")
    # Response is third column (T), drop date and time columns
    y = df.iloc[:, 2].values
    X = pd.get_dummies(df.iloc[:, 3:], drop_first=False).values
    return X, y


def load_bias_correction() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Bias correction of numerical weather prediction.
    Response: Next_Tmax
    """
    df = _load_or_download("BiasCorrection")
    # Remove rows with NA
    df = df.dropna()
    # Drop first column (Date), use Next_Tmax as response, exclude last 2 columns
    y = df.iloc[:, -2].values
    X = pd.get_dummies(df.iloc[:, 2:-2], drop_first=False).values
    return X, y


def load_electrical_stability() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Electrical Grid Stability Simulated Data.
    Response: last column before stability flag
    """
    df = _load_or_download("ElectricalStability")
    y = df.iloc[:, -2].values
    X = pd.get_dummies(df.iloc[:, :-2], drop_first=False).values
    return X, y


def load_gas_turbine() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Gas Turbine CO and NOx Emission Data.
    Response: TEY (Turbine Energy Yield)
    """
    # Gas turbine requires loading multiple years
    dfs = []
    for year in [2011, 2012, 2013, 2014, 2015]:
        url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/00455/gt_{year}.csv"
        filepath = DATA_DIR / "GasTurbine" / f"gt_{year}.csv"
        if not filepath.exists():
            _download_with_retry(url, filepath)
        dfs.append(pd.read_csv(filepath))

    df = pd.concat(dfs, ignore_index=True)
    # Response is column 8 (TEY)
    y = df.iloc[:, 7].values
    X = pd.get_dummies(df.drop(columns=[df.columns[7]]), drop_first=False).values
    return X, y


def load_abalone() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Abalone dataset.
    Response: 9th column (Rings)
    """
    df = _load_or_download("abalone")
    # No header in this file, use column indices
    df.columns = [f"col_{i}" for i in range(len(df.columns))]
    # Response is col_8 (9th column, 0-indexed as 8)
    y = df["col_8"].values
    X = pd.get_dummies(df.drop(columns=["col_8"]), drop_first=False).values
    return X, y


def load_wine_quality_red() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Red Wine Quality dataset.
    Response: 12th column (quality)
    """
    df = _load_or_download("WineQualityRed")
    # No header in original file
    df.columns = [f"col_{i}" for i in range(len(df.columns))]
    # Response is col_11 (12th column)
    y = df["col_11"].values
    X = pd.get_dummies(df.drop(columns=["col_11"]), drop_first=False).values
    return X, y


# Note: The following datasets require special handling or external packages:
# - ResidentialBuilding: Excel file with complex structure
# - LungCancerGenomic: .rda file from R, requires pyreadr
# - StructureActivity: tar file with multiple .dat files
# - BloodBrain: R dataset via caret
# - GSE65904: GEO expression data, requires GEOparse

# These are provided as stubs that would need additional implementation
def load_residential_building() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Residential Building dataset - requires Excel parsing."""
    warnings.warn("load_residential_building not fully implemented")
    raise NotImplementedError("Requires Excel parsing with specific sheet handling")


def load_lung_cancer_genomic() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Lung Cancer Genomic data - requires .rda file parsing."""
    warnings.warn("load_lung_cancer_genomic not fully implemented")
    raise NotImplementedError("Requires pyreadr for .rda file")


def load_structure_activity() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Structure Activity data - requires tar file extraction."""
    warnings.warn("load_structure_activity not fully implemented")
    raise NotImplementedError("Requires tar file extraction")


def load_blood_brain() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Blood Brain Barrier data."""
    # This is available via sklearn but differently formatted
    warnings.warn("load_blood_brain not fully implemented")
    raise NotImplementedError("Requires caret R package data conversion")


def load_gse65904() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """GSE65904 genomic data - requires GEOquery."""
    warnings.warn("load_gse65904 not fully implemented")
    raise NotImplementedError("Requires GEOparse for GEO data")


# Mapping for easy access
REAL_DATA_FUNCTIONS = {
    "boston_housing": load_boston,
    "california_housing": load_california_housing,
    "casp": load_casp,
    "energy": load_energy,
    "air_quality": load_air_quality,
    "bias_correction": load_bias_correction,
    "electrical_stability": load_electrical_stability,
    "gas_turbine": load_gas_turbine,
    "abalone": load_abalone,
    "wine_quality_red": load_wine_quality_red,
}
