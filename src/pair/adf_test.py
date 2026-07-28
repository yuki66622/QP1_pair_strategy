from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd


def adf_test(
    series: pd.Series,
    min_obs: int = 120,
    alpha: float = 0.05,
) -> dict:
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    input_size = len(s)

    if input_size < min_obs:
        return {
            "adf_stat": float("nan"),
            "pvalue": float("nan"),
            "usedlag": None,
            "nobs": input_size,
            "critical_values": None,
            "alpha": alpha,
            "is_stationary": None,
            "status": "insufficient_data",
        }

    if s.nunique() <= 1:
        return {
            "adf_stat": float("nan"),
            "pvalue": float("nan"),
            "usedlag": None,
            "nobs": input_size,
            "critical_values": None,
            "alpha": alpha,
            "is_stationary": None,
            "status": "constant_series",
        }

    adf_stat, pvalue, usedlag, nobs, crit, _ = adfuller(
        s,
        regression="c",
        autolag="AIC",
    )

    return {
        "adf_stat": float(adf_stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {
            level: float(value)
            for level, value in crit.items()
        },
        "alpha": alpha,
        "is_stationary": bool(pvalue < alpha),
        "status": "success",
    }
