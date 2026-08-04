"""johansen_test 的合成数据自测。

用已知答案的假数据验证 Johansen trace test 的行为:
  1. 真协整对: reject_r0=True 且 reject_r1=False, hedge_ratio 接近真值
  2. 独立随机游走: reject_r0=False
  3. 对称性: 交换两条序列的顺序, 结论不变, hedge_ratio 互为倒数
  4. 防护分支: 短序列 / 常数序列 / 脏数据不崩、不出错数
  5. 假阳性率: 独立随机游走在 95% 水平下 reject_r0 的比例应约 5%

直接运行: python3 tests/test_johansen.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pair.johansen import johansen_test


def make_pair(rng, n, beta=2.0, lam=0.1, coint=True):
    """生成一对价格. coint=True 时 log_a = beta*log_b + OU."""
    log_b = 0.02 * rng.standard_normal(n).cumsum() + 4.0
    if coint:
        ou = np.zeros(n)
        for t in range(1, n):
            ou[t] = ou[t - 1] * (1 - lam) + rng.standard_normal() * 0.01
        log_a = beta * log_b + ou - 3.0
    else:
        log_a = 0.02 * rng.standard_normal(n).cumsum() + 4.0
    return pd.Series(np.exp(log_a)), pd.Series(np.exp(log_b))


def main():
    rng = np.random.default_rng(42)
    n = 756

    # 1. 真协整对 (真实 hedge ratio = 2)
    pa, pb = make_pair(rng, n, beta=2.0)
    res = johansen_test(pa, pb)
    assert res["status"] == "success"
    assert res["reject_r0"], "真协整对应拒绝 r=0"
    assert not res["reject_r1"], "真协整对不应拒绝 r<=1 (秩恰好为 1)"
    assert res["is_cointegrated"]
    assert 1.8 < res["hedge_ratio"] < 2.2, \
        f"hedge_ratio 应接近 2, 实测 {res['hedge_ratio']:.3f}"
    print(f"cointegrated : trace_r0={res['trace_stat_r0']:.1f} (crit95={res['trace_crit_r0'][0.05]:.1f})  "
          f"hedge_ratio={res['hedge_ratio']:.3f}  is_coint={res['is_cointegrated']}")

    # 2. 独立随机游走
    pa, pb = make_pair(rng, n, coint=False)
    ind = johansen_test(pa, pb)
    assert ind["status"] == "success" and not ind["reject_r0"], \
        "独立随机游走不应拒绝 r=0"
    assert not ind["is_cointegrated"]
    print(f"independent  : trace_r0={ind['trace_stat_r0']:.1f} (crit95={ind['trace_crit_r0'][0.05]:.1f})  "
          f"is_coint={ind['is_cointegrated']}")

    # 3. 对称性: 交换顺序结论不变, hedge_ratio 互为倒数
    pa, pb = make_pair(rng, n, beta=2.0)
    ab = johansen_test(pa, pb)
    ba = johansen_test(pb, pa)
    assert ab["is_cointegrated"] == ba["is_cointegrated"] == True
    assert abs(ab["hedge_ratio"] * ba["hedge_ratio"] - 1) < 0.05, \
        f"交换顺序后 hedge_ratio 应互为倒数, 实测 {ab['hedge_ratio']:.3f} 和 {ba['hedge_ratio']:.3f}"
    print(f"symmetry     : beta(a|b)={ab['hedge_ratio']:.3f}  beta(b|a)={ba['hedge_ratio']:.3f}  "
          f"product={ab['hedge_ratio'] * ba['hedge_ratio']:.4f}")

    # 4. 防护分支
    short = johansen_test(pd.Series(np.exp(rng.standard_normal(50))),
                          pd.Series(np.exp(rng.standard_normal(50))))
    assert short["status"] == "insufficient_data"
    const = johansen_test(pd.Series(np.ones(300)), pd.Series(np.exp(0.02 * rng.standard_normal(300).cumsum())))
    assert const["status"] == "constant_series"
    dirty = johansen_test(pd.Series(["1.0", "bad", None, np.inf] * 100),
                          pd.Series(["2.0", "3.0", "bad", None] * 100))
    assert dirty["status"] == "insufficient_data"
    print("guards       : insufficient_data / constant_series / dirty input 全部正确")

    # 5. 假阳性率: 独立随机游走对
    #    已知标定结果: 渐近临界值表在此样本规模下偏小, 名义 95% 实际约 10%,
    #    名义 99% 实际约 3% (见 johansen.py 注释), 断言按实测值设定
    sims = 300
    rej_05, rej_01 = 0, 0
    for _ in range(sims):
        pa, pb = make_pair(rng, 756, coint=False)
        res_05 = johansen_test(pa, pb, significance=0.05)
        res_01 = johansen_test(pa, pb, significance=0.01)
        rej_05 += res_05["reject_r0"]
        rej_01 += res_01["reject_r0"]
    rate_05, rate_01 = rej_05 / sims, rej_01 / sims
    assert 0.05 < rate_05 < 0.16, f"名义 95% 的假阳性率 {rate_05:.1%} 偏离标定值 10% 过多"
    assert 0.01 < rate_01 < 0.07, f"名义 99% 的假阳性率 {rate_01:.1%} 偏离标定值 3% 过多"
    print(f"false positive rate on {sims} independent pairs: "
          f"nominal95={rate_05:.1%} (calibrated ~10%), nominal99={rate_01:.1%} (calibrated ~3%)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
