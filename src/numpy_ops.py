"""Three meaningful NumPy operations used by the pipeline (each with its "why")."""
import numpy as np
import pandas as pd


def invalid_value_mask(values: pd.Series, *, allow_zero: bool = False) -> np.ndarray:
    """(1) Boolean mask of non-positive (or negative) values.

    Why: finding impossible prices/weights is one vectorised comparison on the raw
    ndarray - far faster and clearer than a Python loop, and the mask can be reused to
    count, report and null the offending cells consistently.
    """
    arr = values.to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(invalid="ignore"):
        return (arr < 0) if allow_zero else (arr <= 0)


def safe_ratio(num: pd.Series, den: pd.Series) -> np.ndarray:
    """(2a) Vectorised ratio with np.where guarding divide-by-zero / NaN denominators.

    Why: freight_ratio = freight / price must not explode when price is 0 or missing.
    """
    n = num.to_numpy(dtype="float64", na_value=np.nan)
    d = den.to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, n / d, np.nan)


def days_between(later: pd.Series, earlier: pd.Series) -> np.ndarray:
    """(2b) Fractional days between two datetime columns using numpy timedelta64.

    Why: avoids per-row Python datetime objects; NaT propagates to NaN automatically.
    """
    delta = later.to_numpy(dtype="datetime64[ns]") - earlier.to_numpy(dtype="datetime64[ns]")
    return delta / np.timedelta64(1, "D")


def percentile_bounds(values: pd.Series, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """(3) Tail bounds via np.percentile (used for robust plotting ranges).

    Why: price/freight/weight are heavily right-skewed; clipping plot ranges to the
    1st-99th percentile keeps histograms readable without deleting any data.
    """
    arr = values.to_numpy(dtype="float64", na_value=np.nan)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def log1p_transform(x):
    """(4) np.log1p to reduce right skew of monetary/count features (used in the linear model)."""
    return np.log1p(np.clip(x, 0, None))
