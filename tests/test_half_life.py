"""half_life 的合成数据自测。

用已知答案的假数据验证 half-life 估计的行为:
  1. 精度: OU 过程的估计值应接近理论值 ln(2)/-ln(1-lambda)
  2. 随机游走: 估计出的 half-life 应很大 (注意小样本偏差会给出有限值)
  3. 防护分支: 短序列 / 常数序列 / 脏数据不崩、不出错数
  4. 标定: 多条 OU 的估计中位数应落在理论值附近

直接运行: python3 tests/test_half_life.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pair.half_life import half_life


def make_ou(rng, n, lam, sigma=0.1):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 - lam) + rng.standard_normal() * sigma
    return pd.Series(x)


def main():
    rng = np.random.default_rng(42)
    n = 756

    # 1. 精度自测: 两种回归速度的 OU
    for lam in [0.05, 0.15]:
        theory = np.log(2) / -np.log(1 - lam)
        res = half_life(make_ou(rng, n, lam))
        assert res["status"] == "success", f"OU 应成功估计, 实际 {res['status']}"
        assert 0.6 * theory < res["half_life"] < 1.6 * theory, \
            f"lam={lam}: 估计 {res['half_life']:.1f} 天偏离理论值 {theory:.1f} 天过多"
        print(f"OU lam={lam}: half_life={res['half_life']:.1f}d  theory={theory:.1f}d")

    # 2. 随机游走: 真实 half-life 无穷, 但 AR(1) 系数的小样本偏差
    #    (Dickey-Fuller 向下偏) 会给出一个有限的大数 (量级 ~n/5)
    #    这就是 half-life 不能单独当筛选检验的原因, 必须配 ADF/VR
    rw = half_life(pd.Series(rng.standard_normal(n).cumsum()))
    assert rw["status"] in ("success", "no_mean_reversion")
    if rw["status"] == "success":
        assert rw["half_life"] > 50, \
            f"随机游走的估计 half-life 应很大, 实测 {rw['half_life']:.1f}"
    print(f"random walk: half_life={rw['half_life']:.1f}d  status={rw['status']}")

    # 3. 防护分支
    short = half_life(pd.Series(rng.standard_normal(50)))
    assert short["status"] == "insufficient_data"
    const = half_life(pd.Series(np.ones(300)))
    assert const["status"] == "constant_series"
    dirty = half_life(pd.Series(["1.0", "bad", None, np.inf] * 100))
    assert dirty["status"] == "insufficient_data" and dirty["nobs"] == 100
    print("guards     : insufficient_data / constant_series / dirty input 全部正确")

    # 4. 标定: 200 条 OU (lam=0.05, 理论 13.5 天), 中位数应在理论值附近
    theory = np.log(2) / -np.log(0.95)
    hls = [half_life(make_ou(rng, n, 0.05))["half_life"] for _ in range(200)]
    med = float(np.median(hls))
    assert 0.75 * theory < med < 1.35 * theory, \
        f"OU half-life 估计中位数 {med:.1f} 偏离理论值 {theory:.1f} 过多"
    print(f"calibration on 200 OU: median={med:.1f}d  theory={theory:.1f}d")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
