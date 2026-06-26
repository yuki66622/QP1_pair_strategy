import sys
from pathlib import Path
import pandas as pd

# __file__ is the path of current script
# resolve() returns absolute path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.single_stock_filter import single_stock_filter

from src.data.pair_stock_filter import (
    same_sector_filter,
    adv_filter,
    spread_filter,
    top_k_correlation_filter,
    vol_ratio_filter,
)

def main():
    # read data
    universe_raw = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "crsp_daily.csv",
    )

    # filter single stocks first
    eligible_universe = single_stock_filter(universe_raw)

    # generate return matrix before correlation filter
    returns = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "daily_returns.parquet"
)


    # run filters in order
    pairs = same_sector_filter(
        universe=eligible_universe,
        ticker_col="permno",
        sector_col="sector",
    )
    print("After same sector:", len(pairs))

    pairs = adv_filter(
        pairs=pairs,
        universe=eligible_universe,
        ticker_col="permno",
        adv_col="adv_60d",
        min_adv=5_000_000,
    )
    print("After ADV:", len(pairs))

    # only run spread filter if you actually have spread column
    if "rel_spread_60d" in eligible_universe.columns:
        pairs = spread_filter(
            pairs=pairs,
            universe=eligible_universe,
            ticker_col="permno",
            spread_col="rel_spread_60d",
            max_rel_spread=0.002,
        )
        print("After spread:", len(pairs))
    else:
        print("No spread column. Skip spread filter.")

    pairs = top_k_correlation_filter(
        pairs=pairs,
        returns=returns,
        top_k=20,
        min_corr=None,
        checked_data_range=252,
        min_valid_returns=200,
    )
    print("After top-k correlation:", len(pairs))

    pairs = vol_ratio_filter(
        pairs=pairs,
        returns=returns,
        checked_data_range=252,
        min_valid_returns=200,
        min_ratio=0.5,
    )
    print("After vol ratio:", len(pairs))

    # save output
    output_path = PROJECT_ROOT / "data" / "processed" / "candidate_pairs.csv"

    pair_df = pd.DataFrame(pairs, columns=["stock_a", "stock_b"])
    pair_df.to_csv(output_path, index=False)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()