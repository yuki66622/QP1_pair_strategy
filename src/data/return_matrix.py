import pandas as pd

def make_return_matrix(
    df: pd.DataFrame,
    date_col: str = "dlycaldt",
    ticker_col: str = "permno",
    return_col: str = "dlyretx",
) -> pd.DataFrame:
    """
    Convert CRSP long-format return data into wide return matrix.

    Input format:
        permno | ticker | dlycaldt | dlyretx

    Output format:
        date x permno return matrix

    Example:
        index = date
        columns = permno
        values = daily returns
    """
    data = df[[date_col, ticker_col, return_col]].copy()

    # Date must be datetime
    data[date_col] = pd.to_datetime(data[date_col])

    # Use string ID to avoid weird column type issues later
    data[ticker_col] = data[ticker_col].astype(str)

    # CRSP returns may sometimes contain non-numeric missing codes
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")

    returns = data.pivot_table(
        index=date_col,
        columns=ticker_col,
        values=return_col,
        aggfunc="first",
    )

    returns = returns.sort_index()

    return returns