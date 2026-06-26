from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.reshape import make_return_matrix


def main():
    input_path = PROJECT_ROOT / "data" / "raw" / "crsp_daily.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "daily_returns.parquet"

    print("Loading CRSP daily data...")
    df = pd.read_csv(input_path)

    print("Building return matrix...")
    returns = make_return_matrix(
        df=df,
        date_col="dlycaldt",
        ticker_col="permno",
        return_col="dlyretx",
    )

    print(returns.head())
    print(f"Return matrix shape: {returns.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(output_path)

    print(f"Saved return matrix to: {output_path}")


if __name__ == "__main__":
    main()