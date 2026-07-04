import pandas as pd
from itertools import combinations
import numpy as np

def _pair_key(a,b):
    """
    A helper function 

    Avoid redundant computation by enforcing a consistent order

    """
    # sorted() returns list 
    return tuple(sorted((a,b)))

def same_sector_filter(
    universe: pd.DataFrame,
    ticker_col: str="permno",
    sector_col: str="sector",
) -> list[tuple[str,str]]: 
    # [[]] is generic type annotation
    """
    Generate pairs only from the same sector.

    Don't group too detailed here

    """
    pairs=[]

    # keep complete lines
    universe=universe.dropna(subset=[ticker_col,sector_col])

    for sector_name, sector_df in universe.groupby(sector_col):
        ids = sector_df[ticker_col].astype(str).tolist()
        # need at least two stocks to generate a pair
        if len(ids)<2:
            continue
        for a,b in combinations(ids,2):
            # use _pair_key first  
            pairs.append(_pair_key(a,b))
    return list(set(pairs))

def adv_filter(
        pairs:list[tuple[str,str]],
        universe: pd.DataFrame,
        # Non-default parameters must come before default parameters
        ticker_col: str='permno',       
        adv_col: str = "adv_60d",
        min_adv: float = 5_000_000,
)-> list[tuple[str, str]]:
    """
    Make sure both stocks are liquid enough: check the minimum
    """
    # keep data without missing value
    universe = universe.dropna(subset=[ticker_col, adv_col])

    # universe.set_index(ticker_col)['adv_col'] is a series
    # to_dict() converts the Series into {ticker: adv}
    adv_map=universe.set_index(ticker_col)[adv_col].to_dict()

    filtered=[]

    for a,b in pairs:

        # a and b may already been cleaned
        if a not in adv_map or b not in adv_map:
            continue
        
        adv_a=adv_map[a]
        adv_b=adv_map[b]

        if min(adv_a,adv_b)>=min_adv:
            filtered.append((a,b))
    # pairs are assumed to have been normalized by _pair_key upstream 
    # so just preserve the original (a, b) and return filtered directly
    return filtered

def spread_filter(
        pairs:list[tuple[str,str]],
        universe: pd.DataFrame,
        ticker_col: str='permno',  
        # suppose relative spread are given     
        spread_col: str = "rel_spread_60d",
        max_rel_spread: float = 0.002,
)-> list[tuple[str, str]]:
    """
    """
    universe=universe.dropna(subset=[spread_col,ticker_col])

    spread_map=universe.set_index(ticker_col)[spread_col].to_dict()

    filtered=[]

    for a,b in pairs:
        
        if a not in spread_map or b not in spread_map:
            continue   
            
        rel_spread_a=spread_map[a]
        rel_spread_b=spread_map[b]

        if max(rel_spread_a,rel_spread_b) <= max_rel_spread:
            filtered.append((a, b))  
    
    return filtered 

def top_k_correlation_filter(
    pairs:list[tuple[str,str]],
    returns: pd.DataFrame,  
    ticker_col: str='permno',
    top_k: int=20,
    # leave None for now    
    min_corr: float=None,
    checked_data_range: int=252,
    # some stocks may have missing return
    # also better check now since live trading will not use crsp data
    min_valid_returns: int=200,
)-> list[tuple[str, str]]:
    """
    Keep top-k most correlated paris.
    """
    # only keeps data in one years
    returns=returns.tail(checked_data_range)

    # data from return matrix may be int
    returns.columns = returns.columns.astype(str)

    # ?
    pairs = [_pair_key(str(a), str(b)) for a, b in pairs]
    # delete duplicates and set is faster 
    pair_set = set(pairs)

    # extract all stocks used in candidate pairs
    stocks = sorted(set([x for pair in pairs for x in pair]))
    # only keep stocks that have a return column
    stocks = [x for x in stocks if x in returns.columns]

    # need at least two stocks to compute correlation
    if len(stocks) < 2:
        return []

    # return data for candidate stocks only
    ret_sub = returns[stocks]

    # keep stocks with enough non-missing data
    valid_stocks = [
        stock for stock in stocks
        if ret_sub[stock].notna().sum() >= min_valid_returns
    ]

    ret_sub = ret_sub[valid_stocks]

    # compute correlation matrix
    # min_periods means two stocks need at least this many overlapping valid returns
    corr_matrix = ret_sub.corr(min_periods=min_valid_returns)

    selected_pairs = set()

    for stock in valid_stocks:
        # correlation between this stock and all other stocks
        # drop itself because corr(stock, stock) = 1
        corr_series = corr_matrix[stock].drop(index=stock).dropna()

        allowed_neighbors = []

        for other, corr_value in corr_series.items():
            pair = _pair_key(stock, other)

            if pair not in pair_set:
                continue

            if min_corr is not None and corr_value < min_corr:
                continue

            allowed_neighbors.append((other, corr_value))

        # sort neighbors by correlation from high to low
        allowed_neighbors = sorted(
            allowed_neighbors,
            key=lambda x: x[1],
            reverse=True,
        )

        top_neighbors = allowed_neighbors[:top_k]

        for other, corr_value in top_neighbors:
            selected_pairs.add(_pair_key(stock, other))

    return list(selected_pairs)

def vol_ratio_filter(
    pairs: list[tuple[str, str]],
    returns: pd.DataFrame,
    checked_data_range: int = 252,
    min_valid_returns: int = 200,
    min_ratio: float = 0.5,
) -> list[tuple[str, str]]:
    """
    Keep pairs whose volatility levels are not too different.

    Equivalent rule:
        vol_A / vol_B between 0.5 and 2.0
    """

    returns = returns.copy()
    returns.columns = returns.columns.astype(str)
    returns = returns.tail(checked_data_range)

    # daily return volatility for each stock
    vol_std = returns.std(skipna=True)

    # count valid return observations for each stock
    valid_counts = returns.notna().sum()

    filtered = []

    for a, b in pairs:
        a = str(a)
        b = str(b)

        # both stocks must exist in return matrix
        if a not in vol_std.index or b not in vol_std.index:
            continue

        # both stocks must have enough valid returns
        if valid_counts[a] < min_valid_returns or valid_counts[b] < min_valid_returns:
            continue

        vol_a = vol_std[a]
        vol_b = vol_std[b]

        # skip missing or zero volatility
        if pd.isna(vol_a) or pd.isna(vol_b):
            continue

        if vol_a <= 0 or vol_b <= 0:
            continue

        # symmetric volatility ratio
        ratio = min(vol_a, vol_b) / max(vol_a, vol_b)

        if ratio >= min_ratio:
            filtered.append(_pair_key(a, b))

    return list(set(filtered))

