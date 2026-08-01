"""hurst_exponent 的合成数据自测。

用已知答案的假数据验证 Hurst 实现的行为:
  1. 方向: 随机游走 H~0.5, OU 过程 H<0.5, 趋势序列 H>0.5
  2. 防护分支: 短序列 / 常数序列 / 脏数据不崩、不出错数
  3. 标定: 多条独立随机游走的 H 均值应接近 0.5

直接运行: python3 tests/test_hurst.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pair.hurst import hurst_exponent


def main():
    rng = np.random.default_rng(42)
    n = 756

    # 1. 方向自测
    rw = hurst_exponent(pd.Series(rng.standard_normal(n).cumsum()))
    assert rw["status"] == "success" and 0.4 < rw["hurst"] < 0.6, \
        f"随机游走 H 应接近 0.5, 实测 {rw['hurst']:.3f}"
    print(f"random walk : H={rw['hurst']:.3f}  mean_reverting={rw['is_mean_reverting']}")

    # OU 过程: lambda=0.15, half-life ~4.3 天, 落在 lag 窗口 (2-20 天) 内
    # 注意: half-life 远大于 max_lag 的均值回归 Hurst 检测不出来 (短 lag 下扩散
    # 与随机游走无异), 例如 lam=0.05 (half-life ~14 天) 实测 H~0.44
    lam = 0.15
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 - lam) + rng.standard_normal() * 0.1
    ou = hurst_exponent(pd.Series(x))
    assert ou["status"] == "success" and ou["hurst"] < 0.4, \
        f"OU 过程 H 应明显小于 0.5, 实测 {ou['hurst']:.3f}"
    assert ou["is_mean_reverting"], "OU 过程应判为均值回归"
    print(f"OU process  : H={ou['hurst']:.3f}  mean_reverting={ou['is_mean_reverting']}")

    # 持续性序列: 增量为正自相关的 AR(0.9), 模拟动量/趋势行为
    # 注意: 确定性线性趋势测不出来 (std 对差分做了居中, 常数漂移被消掉),
    # 这个估计量对 drift 免疫, 测的是随机持续性
    eps = rng.standard_normal(n)
    inc = np.zeros(n)
    for t in range(1, n):
        inc[t] = 0.9 * inc[t - 1] + eps[t]
    trend = hurst_exponent(pd.Series(inc.cumsum()))
    assert trend["status"] == "success" and trend["hurst"] > 0.7, \
        f"持续性序列 H 应明显大于 0.5, 实测 {trend['hurst']:.3f}"
    print(f"persistent  : H={trend['hurst']:.3f}  mean_reverting={trend['is_mean_reverting']}")

    # 2. 防护分支
    short = hurst_exponent(pd.Series(rng.standard_normal(50)))
    assert short["status"] == "insufficient_data"
    const = hurst_exponent(pd.Series(np.ones(300)))
    assert const["status"] == "constant_series"
    dirty = hurst_exponent(pd.Series(["1.0", "bad", None, np.inf] * 100))
    assert dirty["status"] == "insufficient_data" and dirty["nobs"] == 100
    print("guards      : insufficient_data / constant_series / dirty input 全部正确")

    # 3. 标定: 200 条独立随机游走的 H 均值应接近 0.5
    sims = 200
    hs = [
        hurst_exponent(pd.Series(rng.standard_normal(n).cumsum()))["hurst"]
        for _ in range(sims)
    ]
    mean_h = float(np.mean(hs))
    assert 0.45 < mean_h < 0.55, f"随机游走 H 均值 {mean_h:.3f} 偏离 0.5 过多"
    print(f"calibration on {sims} random walks: mean H={mean_h:.3f} (expect ~0.5)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
