import numpy as np
import pandas as pd


def hurst_exponent(
    series: pd.Series,
    min_obs: int = 120,
    max_lag: int = 20,
    threshold: float = 0.5,
) -> dict:
    # 输入是价格水平 (spread), 不是收益率
    # 原理: Var(x_{t+tau} - x_t) ~ tau^{2H}, 即 std ~ tau^H
    # 对 log(std) ~ log(tau) 回归, 斜率就是 H
    # H < 0.5 均值回归, H = 0.5 随机游走, H > 0.5 趋势
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    input_size = len(s)

    if input_size < min_obs:
        return {
            "hurst": float("nan"),
            "n_lags": 0,
            "nobs": input_size,
            "threshold": threshold,
            "is_mean_reverting": None,
            "status": "insufficient_data",
        }

    if s.nunique() <= 1:
        return {
            "hurst": float("nan"),
            "n_lags": 0,
            "nobs": input_size,
            "threshold": threshold,
            "is_mean_reverting": None,
            "status": "constant_series",
        }

    x = s.to_numpy(dtype=float)

    # lag 上限不超过样本的 1/4, 保证每个 lag 有足够的差分对
    # 默认 max_lag=20: 合成数据标定显示大 lag 下估计量向下偏且噪声大
    # (756 天样本, max_lag=100 时随机游走 mean H=0.46/std=0.07; max_lag=20 时 0.49/0.04)
    max_lag_eff = min(max_lag, input_size // 4)
    lags = np.arange(2, max_lag_eff + 1)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])

    # 剔除 std 为 0 的 lag (局部常数段), 避免 log(0)
    valid = tau > 0
    if valid.sum() < 10:
        return {
            "hurst": float("nan"),
            "n_lags": int(valid.sum()),
            "nobs": input_size,
            "threshold": threshold,
            "is_mean_reverting": None,
            "status": "degenerate_series",
        }

    hurst = float(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0])

    return {
        "hurst": hurst,
        "n_lags": int(valid.sum()),
        "nobs": input_size,
        "threshold": threshold,
        "is_mean_reverting": bool(hurst < threshold),
        "status": "success",
    }
