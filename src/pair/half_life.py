import numpy as np
import pandas as pd


def half_life(
    series: pd.Series,
    min_obs: int = 120,
) -> dict:
    # 输入是价格水平 (spread)
    # 原理: AR(1) 回归 delta_s_t = a + b * s_{t-1} + eps
    # b < 0 表示均值回归, phi = 1 + b 是日衰减系数
    # half_life = ln(2) / -ln(phi), 单位是天
    #
    # 注意: 这里不对 b 做显著性检验 -- b 的 t 统计量服从的是
    # Dickey-Fuller 分布而不是 t 分布 (这本质上就是 ADF 检验),
    # 显著性判断交给 adf_test / variance_ratio, 本模块只负责量化速度
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    input_size = len(s)

    def _fail(status):
        return {
            "half_life": float("nan"),
            "b": float("nan"),
            "phi": float("nan"),
            "nobs": input_size,
            "status": status,
        }

    if input_size < min_obs:
        return _fail("insufficient_data")

    if s.nunique() <= 1:
        return _fail("constant_series")

    x = s.to_numpy(dtype=float)
    lag = x[:-1]
    diff = np.diff(x)

    b, a = np.polyfit(lag, diff, 1)
    phi = 1.0 + b

    if b >= 0:
        # 没有均值回归 (随机游走或发散), half-life 无定义
        return {
            "half_life": float("inf"),
            "b": float(b),
            "phi": float(phi),
            "nobs": input_size,
            "status": "no_mean_reversion",
        }

    if phi <= 0:
        # phi <= 0 意味着逐日反号震荡, 不是价差序列的正常动态
        return _fail("oscillatory")

    hl = float(np.log(2) / -np.log(phi))

    return {
        "half_life": hl,
        "b": float(b),
        "phi": float(phi),
        "nobs": input_size,
        "status": "success",
    }
