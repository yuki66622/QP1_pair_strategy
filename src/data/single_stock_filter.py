import pandas as pd

def single_stock_filter(
        df:pd.DataFrame,
        min_price: float=5.0,
        min_ADV_60D: float=5_000_000,
        min_valid_days: int=719
)-> pd.DataFrame :
    """
    single-stock filters

    Input:
    df: stock summary table
    min_price: mininum price 
    min_ADV_60D: adv based on 60 days
    min_valid_days: 756 trading days(3 year) x 95% = 718.2 (missing data ratio < 5%)

    Output:
    eligible stock table after first test
    """
    # copy the table to aviod the original infos 
    filtered = df.copy()

    # 1. common stock only
    # SHRCD = Share Code
    # 10, 11 -> common stock
    # iltered["shrcd"].isin([10, 11]) generates True/False table
    filtered = filtered[filtered["shrcd"].isin([10, 11])]

    # 2. NYSE/AMEX/NASDAQ only
    # EXCHCD = Exchange Code
    # 1 = NYSE; 2 = AMEX; 3 = NASDAQ
    filtered=filtered[filtered['exchcd'].isin([1,2,3])]

    # 3. minnum price
    filtered=filtered[filtered["latest_price"].abs() > min_price]

    # 4. ADV 
    filtered=filtered[filtered["adv_60d"] > min_ADV_60D]

    # 5. history
    filtered = filtered[filtered["valid_days_756"] >= min_valid_days]

    # reindex
    return filtered.reset_index(drop=True)


