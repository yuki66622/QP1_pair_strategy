from pathlib import Path
import pandas as pd
import wrds


def read_ticker_list(path: str) -> list[str]:
    """
    Read ticker list from a txt file.
    One ticker per line.
    """
    with open(path, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    return sorted(set(tickers))


def sql_quote(x: str) -> str:
    """
    Safely quote ticker for SQL IN clause.
    """
    return "'" + x.replace("'", "''") + "'"


def download_crsp_daily(
    ticker_file: str,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    output_file: str = "data/raw/crsp_daily.csv",
) -> pd.DataFrame:
    """
    Download CRSP daily stock data from WRDS.

    Source:
        crsp_a_stock.wrds_dsfv2_query

    Output:
        One row per ticker per trading day.
    """
    tickers = read_ticker_list(ticker_file)

    if len(tickers) == 0:
        raise ValueError("Ticker list is empty.")

    ticker_sql = ", ".join(sql_quote(t) for t in tickers)

    query = f"""
        SELECT
            permno,
            ticker,
            permco,
            siccd,
            securitynm,
            dlycaldt,
            dlyprc,
            dlyret,
            dlyretx,
            dlyvol,
            dlybid,
            dlyask,
            dlyprcvol,
            dlycap,
            shrout
        FROM crsp_a_stock.wrds_dsfv2_query
        WHERE dlycaldt BETWEEN '{start_date}' AND '{end_date}'
          AND ticker IN ({ticker_sql})
        ORDER BY ticker, dlycaldt
    """

    print(f"Connecting to WRDS...")
    conn = wrds.Connection()

    print(f"Downloading CRSP daily data for {len(tickers)} tickers...")
    df = conn.raw_sql(query, date_cols=["dlycaldt"])

    conn.close()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    download_crsp_daily(
        ticker_file="data/sp500_tickers.txt",
        start_date="2023-01-01",
        end_date="2025-12-31",
        output_file="data/raw/crsp_daily.csv",
    )
    