from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd

def get_engle_granger_pvalue(price_a, price_b):
    df = pd.concat([price_a, price_b], axis=1).dropna()

    df = df[df.iloc[:, 0] > 0]
    df = df[df.iloc[:, 1] > 0]

    log_a = np.log(df.iloc[:, 0])
    log_b = np.log(df.iloc[:, 1])

    score, pvalue, critical_values = coint(log_a, log_b)

    return pvalue