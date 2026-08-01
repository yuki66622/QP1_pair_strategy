import numpy as np
import pandas as pd
from scipy.stats import norm


def variance_ratio(
    series: pd.Series,
    q: int = 10,
    min_obs: int = 120,
    alpha: float = 0.05,
) -> dict:
    # 输入是价格水平 (spread), 不是收益率, 内部自己做差分
    # Lo-MacKinlay (1988): VR(q) = Var(q 天差分) / (q * Var(1 天差分))
    # 随机游走 VR=1, 均值回归 VR<1, 动量 VR>1
    # z_robust 对异方差稳健, 金融序列判显著性用它; z_homo 仅供对照
    # q 要和预期的均值回归时间尺度匹配 (量级上取 half-life 附近)
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    input_size = len(s)

    def _fail(status):
        return {
            "vr": float("nan"),
            "z_homo": float("nan"),
            "z_robust": float("nan"),
            "pvalue_homo": float("nan"),
            "pvalue_robust": float("nan"),
            "q": q,
            "nobs": input_size,
            "alpha": alpha,
            "is_mean_reverting": None,
            "status": status,
        }

    if input_size < min_obs:
        return _fail("insufficient_data")

    if s.nunique() <= 1:
        return _fail("constant_series")

    if q < 2 or q > input_size // 4:
        return _fail("invalid_q")

    x = s.to_numpy(dtype=float)
    n = len(x) - 1                      # 1 天差分的个数
    r = np.diff(x)
    mu = (x[-1] - x[0]) / n

    e = r - mu
    sigma2_1 = np.sum(e**2) / (n - 1)

    if sigma2_1 <= 0:
        return _fail("degenerate_series")

    # q 天重叠差分, 带 Lo-MacKinlay 偏差修正的分母 m
    # 注意 m 里已含 q 因子, sigma2_q 直接是"每期方差"的估计, VR 不能再除 q
    y = x[q:] - x[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    sigma2_q = np.sum((y - q * mu) ** 2) / m

    vr = sigma2_q / sigma2_1

    # 同方差假设下的渐近方差
    var_homo = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
    z_homo = (vr - 1) / np.sqrt(var_homo)

    # 异方差稳健版 (Lo-MacKinlay 1988, eq. 24-25)
    e2 = e**2
    denom = np.sum(e2) ** 2
    theta = 0.0
    for j in range(1, q):
        delta_j = np.sum(e2[j:] * e2[:-j]) / denom
        theta += (2 * (q - j) / q) ** 2 * delta_j

    if theta <= 0:
        return _fail("degenerate_series")

    z_robust = (vr - 1) / np.sqrt(theta)

    pvalue_homo = float(2 * (1 - norm.cdf(abs(z_homo))))
    pvalue_robust = float(2 * (1 - norm.cdf(abs(z_robust))))

    return {
        "vr": float(vr),
        "z_homo": float(z_homo),
        "z_robust": float(z_robust),
        "pvalue_homo": pvalue_homo,
        "pvalue_robust": pvalue_robust,
        "q": q,
        "nobs": input_size,
        "alpha": alpha,
        "is_mean_reverting": bool(vr < 1 and pvalue_robust < alpha),
        "status": "success",
    }
