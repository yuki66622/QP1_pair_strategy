from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "yahoo" / "manual"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DAILY_OUTPUT = PROCESSED_DIR / "yahoo_to_crsp_format.csv"
UNIVERSE_OUTPUT = PROCESSED_DIR / "yahoo_crsp_universe.csv"


SECTOR_MAP = {
    "NVDA": "Technology",
    "INTC": "Technology",
    "AAPL": "Technology",

    "AMZN": "Internet",
    "GOOGL": "Internet",
    "META": "Internet",

    "GS": "Financials",
    "JPM": "Financials",

    "FDX": "Transportation",
    "FDXF": "Transportation",

    "FOX": "Media",
    "FOXA": "Media",

    "GE": "Industrials",
    "GEHC": "Industrials",
    "GEV": "Industrials",
}


def clean_one_yahoo_file(file_path: Path) -> pd.DataFrame:
    ticker = file_path.stem.upper()

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    df = df.rename(columns={
        "Date": "dlycaldt",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "vol",
    })

    required_cols = [
        "dlycaldt",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "vol",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing columns {missing_cols}")

    df = df[required_cols].copy()

    # Convert date
    df["dlycaldt"] = pd.to_datetime(df["dlycaldt"], errors="coerce")

    # Convert numbers
    numeric_cols = ["open", "high", "low", "close", "adj_close", "vol"]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop dividend/split/event rows and invalid rows
    df = df.dropna(subset=required_cols)

    df = df.sort_values("dlycaldt")

    # CRSP-like identifiers
    # For MVP, use ticker as permno.
    df["ticker"] = ticker
    df["permno"] = ticker

    # CRSP-style price field
    df["prc"] = df["close"]

    # CRSP-style daily return field
    # Use adjusted close return.
    df["dlyretx"] = df["adj_close"].pct_change()

    # Liquidity field
    df["dollar_volume"] = df["close"] * df["vol"]

    df["source"] = "yahoo_manual"

    df = df[
        [
            "permno",
            "ticker",
            "dlycaldt",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "prc",
            "vol",
            "dlyretx",
            "dollar_volume",
            "source",
        ]
    ]

    return df


def build_universe_summary(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["dlycaldt"] = pd.to_datetime(daily["dlycaldt"])
    daily = daily.sort_values(["permno", "dlycaldt"])

    latest = (
        daily.groupby("permno")
        .tail(1)[["permno", "ticker", "prc"]]
        .rename(columns={"prc": "latest_price"})
    )

    last_60 = daily.groupby("permno").tail(60)

    adv_60d = (
        last_60.groupby("permno")["dollar_volume"]
        .mean()
        .reset_index()
        .rename(columns={"dollar_volume": "adv_60d"})
    )

    last_756 = daily.groupby("permno").tail(756)

    valid_days_756 = (
        last_756.groupby("permno")["adj_close"]
        .count()
        .reset_index()
        .rename(columns={"adj_close": "valid_days_756"})
    )

    universe = latest.merge(adv_60d, on="permno", how="inner")
    universe = universe.merge(valid_days_756, on="permno", how="inner")

    # Dummy CRSP fields for MVP.
    # Your manual Yahoo files are already selected large/common-like stocks.
    universe["shrcd"] = 10
    universe["exchcd"] = 3

    # Manual sector mapping
    universe["sector"] = universe["ticker"].map(SECTOR_MAP).fillna("UNKNOWN")

    # Yahoo historical CSV does not contain bid-ask spread.
    # Keep this column as NaN so you remember spread_filter should be skipped.
    universe["rel_spread_60d"] = np.nan

    universe = universe[
        [
            "permno",
            "ticker",
            "sector",
            "shrcd",
            "exchcd",
            "latest_price",
            "adv_60d",
            "valid_days_756",
            "rel_spread_60d",
        ]
    ]

    return universe


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No Yahoo CSV files found in {RAW_DIR}. "
            "Put files like AAPL.csv, MSFT.csv there first."
        )

    all_daily = []

    for file_path in csv_files:
        try:
            temp = clean_one_yahoo_file(file_path)
            print(f"[OK] {file_path.name}: {len(temp)} rows")
            all_daily.append(temp)
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    if not all_daily:
        raise ValueError("No valid Yahoo files processed.")

    daily = pd.concat(all_daily, ignore_index=True)
    daily = daily.sort_values(["dlycaldt", "permno"])

    universe = build_universe_summary(daily)

    daily.to_csv(DAILY_OUTPUT, index=False)
    universe.to_csv(UNIVERSE_OUTPUT, index=False)

    print(f"\nSaved daily CRSP-like data to: {DAILY_OUTPUT}")
    print(daily.head())
    print("\nDaily shape:", daily.shape)

    print(f"\nSaved universe summary to: {UNIVERSE_OUTPUT}")
    print(universe)
    print("\nUniverse shape:", universe.shape)


if __name__ == "__main__":
    main()