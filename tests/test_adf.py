"""adf_test 的合成数据自测。

用已知答案的假数据验证 ADF 实现的行为:
  1. 方向: 白噪声/OU 过程应判平稳, 随机游走应判不平稳
  2. 防护分支: 短序列 / 常数序列 / 脏数据不崩、不出错数
  3. 假阳性率: 独立随机游走在 p<0.05 下的拒绝率应约 5% (检验 size 是否准确)

直接运行: python3 tests/test_adf.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pair.adf_test import adf_test


def main():
    rng = np.random.default_rng(42)

    # 1. 方向自测
    wn = adf_test(pd.Series(rng.standard_normal(500)))
    assert wn["status"] == "success" and wn["is_stationary"], "白噪声应判平稳"
    print(f"white noise : p={wn['pvalue']:.2e}  stationary={wn['is_stationary']}")

    rw = adf_test(pd.Series(rng.standard_normal(500).cumsum()))
    assert rw["status"] == "success" and not rw["is_stationary"], "随机游走应判不平稳"
    print(f"random walk : p={rw['pvalue']:.3f}  stationary={rw['is_stationary']}")

    # OU 过程: lambda=0.05, 理论 half-life = ln2/0.05 ~ 13.9 天, 模拟均值回归的 spread
    lam, n = 0.05, 756
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 - lam) + rng.standard_normal() * 0.1
    ou = adf_test(pd.Series(x))
    assert ou["status"] == "success" and ou["is_stationary"], "OU 过程应判平稳"
    print(f"OU process  : p={ou['pvalue']:.2e}  stationary={ou['is_stationary']}")

    # 2. 防护分支
    short = adf_test(pd.Series(rng.standard_normal(50)))
    assert short["status"] == "insufficient_data"
    const = adf_test(pd.Series(np.ones(300)))
    assert const["status"] == "constant_series"
    dirty = adf_test(pd.Series(["1.0", "bad", None, np.inf] * 100))
    assert dirty["status"] == "insufficient_data" and dirty["nobs"] == 100
    print("guards      : insufficient_data / constant_series / dirty input 全部正确")

    # 3. 假阳性率标定 (500 条独立随机游走, 期望约 5%)
    sims = 500
    rejects = sum(
        adf_test(pd.Series(rng.standard_normal(300).cumsum()))["is_stationary"]
        for _ in range(sims)
    )
    rate = rejects / sims
    assert 0.02 < rate < 0.09, f"假阳性率 {rate:.1%} 偏离 5% 过多, 实现可能有系统性偏差"
    print(f"false positive rate on {sims} random walks: {rate:.1%} (expect ~5%)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
