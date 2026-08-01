"""variance_ratio 的合成数据自测。

用已知答案的假数据验证 Lo-MacKinlay 变差比检验的行为:
  1. 方向: 随机游走 VR~1 不显著, OU 过程 VR<1 显著, 持续性序列 VR>1 显著
  2. 防护分支: 短序列 / 常数序列 / 非法 q / 脏数据不崩、不出错数
  3. 假阳性率: 独立随机游走在 alpha=0.05 下 z_robust 的拒绝率应约 5%
  4. 异方差场景: 波动率聚集的随机游走下 z_robust 的拒绝率仍应约 5%

直接运行: python3 tests/test_variance_ratio.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pair.variance_ratio import variance_ratio


def main():
    rng = np.random.default_rng(42)
    n = 756

    # 1. 方向自测
    rw = variance_ratio(pd.Series(rng.standard_normal(n).cumsum()))
    assert rw["status"] == "success" and 0.7 < rw["vr"] < 1.3, \
        f"随机游走 VR 应接近 1, 实测 {rw['vr']:.3f}"
    assert not rw["is_mean_reverting"], "随机游走不应判为均值回归"
    print(f"random walk : VR={rw['vr']:.3f}  p_robust={rw['pvalue_robust']:.3f}  mr={rw['is_mean_reverting']}")

    # OU 过程: lambda=0.15, half-life ~4.3 天, q=10 覆盖均值回归时间尺度
    lam = 0.15
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 - lam) + rng.standard_normal() * 0.1
    ou = variance_ratio(pd.Series(x))
    assert ou["status"] == "success" and ou["vr"] < 0.8, \
        f"OU 过程 VR 应明显小于 1, 实测 {ou['vr']:.3f}"
    assert ou["is_mean_reverting"], "OU 过程应判为均值回归且显著"
    print(f"OU process  : VR={ou['vr']:.3f}  p_robust={ou['pvalue_robust']:.2e}  mr={ou['is_mean_reverting']}")

    # 持续性序列: 增量为正自相关的 AR(0.9), VR 应大于 1 且显著
    eps = rng.standard_normal(n)
    inc = np.zeros(n)
    for t in range(1, n):
        inc[t] = 0.9 * inc[t - 1] + eps[t]
    pers = variance_ratio(pd.Series(inc.cumsum()))
    assert pers["status"] == "success" and pers["vr"] > 1.5, \
        f"持续性序列 VR 应明显大于 1, 实测 {pers['vr']:.3f}"
    assert not pers["is_mean_reverting"], "持续性序列不应判为均值回归"
    print(f"persistent  : VR={pers['vr']:.3f}  p_robust={pers['pvalue_robust']:.2e}  mr={pers['is_mean_reverting']}")

    # 2. 防护分支
    short = variance_ratio(pd.Series(rng.standard_normal(50)))
    assert short["status"] == "insufficient_data"
    const = variance_ratio(pd.Series(np.ones(300)))
    assert const["status"] == "constant_series"
    bad_q = variance_ratio(pd.Series(rng.standard_normal(n).cumsum()), q=300)
    assert bad_q["status"] == "invalid_q"
    dirty = variance_ratio(pd.Series(["1.0", "bad", None, np.inf] * 100))
    assert dirty["status"] == "insufficient_data" and dirty["nobs"] == 100
    print("guards      : insufficient_data / constant_series / invalid_q / dirty input 全部正确")

    # 3. 假阳性率标定: 同方差随机游走
    sims = 500
    rejects = sum(
        variance_ratio(pd.Series(rng.standard_normal(300).cumsum()))["pvalue_robust"] < 0.05
        for _ in range(sims)
    )
    rate = rejects / sims
    assert 0.02 < rate < 0.09, f"假阳性率 {rate:.1%} 偏离 5% 过多"
    print(f"false positive rate on {sims} random walks: {rate:.1%} (expect ~5%)")

    # 4. 异方差场景: 波动率聚集 (log-vol AR(0.95)) 的随机游走
    #    z_robust 的拒绝率应仍在 5% 附近, 这是它存在的意义
    rej_homo, rej_robust = 0, 0
    for _ in range(sims):
        h = np.zeros(300)
        for t in range(1, 300):
            h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
        inc = rng.standard_normal(300) * np.exp(h)
        res = variance_ratio(pd.Series(inc.cumsum()))
        rej_homo += res["pvalue_homo"] < 0.05
        rej_robust += res["pvalue_robust"] < 0.05
    rate_homo, rate_robust = rej_homo / sims, rej_robust / sims
    assert 0.02 < rate_robust < 0.10, \
        f"异方差下 z_robust 假阳性率 {rate_robust:.1%} 偏离 5% 过多"
    print(f"heteroskedastic RW: homo rejects {rate_homo:.1%}, robust rejects {rate_robust:.1%} (robust expect ~5%)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
