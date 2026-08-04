import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# statsmodels 的 Johansen 只给三档临界值, 没有连续 p-value
# significance -> 临界值表的列索引
_SIG_COL = {0.10: 0, 0.05: 1, 0.01: 2}


def johansen_test(
    price_a: pd.Series,
    price_b: pd.Series,
    min_obs: int = 120,
    k_ar_diff: int = 1,
    significance: float = 0.01,
) -> dict:
    # 输入是原始价格, 内部取对数, 与 get_engle_granger_pvalue 的约定一致
    # 检验逻辑 (trace test, 两资产):
    #   reject_r0: 拒绝"没有协整关系" -- 想要 True
    #   reject_r1: 拒绝"至多一个协整关系" -- 想要 False (True 意味着两条
    #              序列各自平稳, 不是随机游走间的协整, 不是我们要的结构)
    # is_cointegrated = reject_r0 and not reject_r1, 即秩恰好为 1
    #
    # 与 Engle-Granger 的分工: Johansen 对称 (不依赖回归方向), 且协整
    # 向量直接给出 hedge_ratio; EG 有连续 p-value, 便于排序和多重检验校正
    #
    # 重要标定结果 (2026-08-04, 合成数据 1000 对独立随机游走, n=756):
    # 渐近临界值表在这个样本规模下系统性偏小, 名义水平 != 真实假阳性率:
    #   按 95% 临界值判 -> 实际假阳性率约 10%
    #   按 99% 临界值判 -> 实际假阳性率约 3%
    # 因此 significance 默认 0.01; 用 0.05 时要知道它其实是 "90% 置信"
    if significance not in _SIG_COL:
        raise ValueError(f"significance must be one of {sorted(_SIG_COL)}")

    df = pd.concat([price_a, price_b], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    df = df[(df.iloc[:, 0] > 0) & (df.iloc[:, 1] > 0)]

    input_size = len(df)

    def _fail(status):
        return {
            "trace_stat_r0": float("nan"),
            "trace_stat_r1": float("nan"),
            "trace_crit_r0": None,
            "trace_crit_r1": None,
            "reject_r0": None,
            "reject_r1": None,
            "is_cointegrated": None,
            "hedge_ratio": float("nan"),
            "eigenvector": None,
            "significance": significance,
            "nobs": input_size,
            "status": status,
        }

    if input_size < min_obs:
        return _fail("insufficient_data")

    log_prices = np.log(df.to_numpy(dtype=float))

    if np.unique(log_prices[:, 0]).size <= 1 or np.unique(log_prices[:, 1]).size <= 1:
        return _fail("constant_series")

    # det_order=0: 协整关系中含常数项 (spread 均值不必为零), 无时间趋势
    res = coint_johansen(log_prices, det_order=0, k_ar_diff=k_ar_diff)

    col = _SIG_COL[significance]
    trace_stat_r0 = float(res.lr1[0])
    trace_stat_r1 = float(res.lr1[1])
    crit_r0 = {0.10: float(res.cvt[0, 0]), 0.05: float(res.cvt[0, 1]), 0.01: float(res.cvt[0, 2])}
    crit_r1 = {0.10: float(res.cvt[1, 0]), 0.05: float(res.cvt[1, 1]), 0.01: float(res.cvt[1, 2])}
    reject_r0 = bool(trace_stat_r0 > res.cvt[0, col])
    reject_r1 = bool(trace_stat_r1 > res.cvt[1, col])

    # 第一列特征向量是主协整关系 [v0, v1]: v0*log_a + v1*log_b 平稳
    # 归一化成 spread = log_a - hedge_ratio * log_b
    v = res.evec[:, 0]
    hedge_ratio = float(-v[1] / v[0]) if v[0] != 0 else float("nan")

    return {
        "trace_stat_r0": trace_stat_r0,
        "trace_stat_r1": trace_stat_r1,
        "trace_crit_r0": crit_r0,
        "trace_crit_r1": crit_r1,
        "reject_r0": reject_r0,
        "reject_r1": reject_r1,
        "is_cointegrated": bool(reject_r0 and not reject_r1),
        "hedge_ratio": hedge_ratio,
        "eigenvector": [float(v[0]), float(v[1])],
        "significance": significance,
        "nobs": input_size,
        "status": "success",
    }
