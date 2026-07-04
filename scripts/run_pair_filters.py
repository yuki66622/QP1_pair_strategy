import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.data.single_stock_filter import single_stock_filter
from src.data.return_matrix import make_return_matrix
from src.data.pair_stock_filter import (
    same_sector_filter,
    adv_filter,
    top_k_correlation_filter,
    vol_ratio_filter,
)

universe = pd.read_csv("data/processed/yahoo_crsp_universe.csv")
daily = pd.read_csv("data/processed/yahoo_to_crsp.csv")

returns = make_return_matrix(
    daily,
    date_col="dlycaldt",
    ticker_col="permno",
    return_col="dlyretx",
)

eligible = single_stock_filter(
    universe,
    min_price=5.0,
    min_ADV_60D=1_000_000,
    min_valid_days=150,
)

pairs = same_sector_filter(
    eligible,
    ticker_col="permno",
    sector_col="sector",
)

pairs = adv_filter(
    pairs,
    eligible,
    ticker_col="permno",
    adv_col="adv_60d",
    min_adv=1_000_000,
)

# Skip spread_filter for Yahoo MVP
# Yahoo historical CSV does not have bid-ask spread

pairs = top_k_correlation_filter(
    pairs,
    returns,
    top_k=5,
    min_corr=0.2,
    checked_data_range=252,
    min_valid_returns=100,
)

pairs = vol_ratio_filter(
    pairs,
    returns,
    checked_data_range=252,
    min_valid_returns=100,
    min_ratio=0.3,
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "candidate_pairs.csv"

pair_df = pd.DataFrame(pairs, columns=["ticker_a", "ticker_b"])
pair_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved candidate pairs to: {OUTPUT_PATH}")
print(pair_df)
