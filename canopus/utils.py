from canopus.secrets import ALPHA_VANTAGE
from datetime import date, timedelta, datetime

import pandas as pd
import os, logging
import json
import uuid

from iexfinance.stocks import get_historical_data


def get_data_info():
    """
    Reads in a json file that is a mapping between tickers and their files names
    """

    path = os.path.join(os.path.dirname(__file__), "data", os.environ["ENV_TYPE"], "info.json")
    if not os.path.exists(path):
        # need to create an empty file
        return {}

    # we are good to go, the file exists
    with open(path, "r") as f:
        return json.load(f)

def save_data_info(path_dict):
    """
    Save the information about ticker and their file IDs
    """
    path = os.path.join(os.path.dirname(__file__), "data", os.environ["ENV_TYPE"], "info.json")
    with open(path, "w") as f:
        json.dump(path_dict, f)

# for non-float
col_types = {}

def _check_table_health(con, table, cols):
    cur = con.cursor()

    # check for table
    query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME ='{table}';"
    cur.execute(query)    
    fetched = cur.fetchall()
    if not fetched:
        col_str = ""
        # get all the appropriate column types and build string for query
        for col in cols:
            type_ = col_types.get(col, "FLOAT")   # default to FLOAT type
            col_str += f", {col} {type_}"

        # create the table
        query = f"CREATE TABLE {table} (time TIMESTAMP NOT NULL{col_str}, PRIMARY KEY(time));"
        cur.execute(query)  
        con.commit()
        logging.info(f"created missing table for {table}")
        missing_cols = cols
    else:
        missing_cols = []
        # check for price column
        columns = [x[3] for x in fetched]
        logging.info(columns)
        for col in cols:
            if col not in columns:
                # column doesn't exist in table
                query = f"ALTER TABLE {table} ADD {col} float;"
                cur.execute(query)
                missing_cols.append(col)
        logging.info(f"Adding missing columns: {', '.join(missing_cols)}")

    cur.close()
    return missing_cols

def _pull_col_info(con, table, start, end, col):
    return pd.DataFrame()

def pull_data_sql(con, table, start, end, cols=["price"], debug=False):
    """ Try to make cols a list so you can pull multiple attr if needed
    
    parameter:
    con: the psycopg2 connection to the db
    table: most likely the ticker 
    start: time at which data should start
    end: time at which data should end
    cols: the different data we need (e.g. price, sma, )

    returns:
    dataframe
    """
    cur = con.cursor()
    table = table.lower()
    
    # missing_cols is the data you need to pull in full
    missing_cols = _check_table_health(con, table, cols)
    final_df = pd.DataFrame()
    for col in missing_cols:
        col_data =  _pull_col_info(con, table, start, end, col)
        final_df = final_df.merge(col_data, how='outer', on='time')
    
    # push the df to the sql DB
    final_df.to_sql(table, con)
    return final_df

    # check for price column

    # do the actual query
    # query = f"SELECT price FROM {ticker}"
    # cur.execute(query)
    # fetched = cur.fetchall()
    # return fetched

def pull_historical_data(ticker, start, end, debug=False):
    """ If there are gaps between local data and what the users need,
    we will fill in those gaps
    TODO:    needs extensive testing
    TODO:    needs to handle when start or end include nontrading days
    """
    if not isinstance(start, pd.Timestamp):
        raise ValueError("start time has to be a pandas Timestamp")
    if not isinstance(end, pd.Timestamp):
        raise ValueError("end time has to be a pandas Timestamp")
    
    path_dict = get_data_info()

    if ticker not in path_dict:
        # doesnt exist yet, can just read in the whole thing and save it
        print(f"Got {ticker} info from IEX")
        file_loc = os.path.join(os.path.dirname(__file__), "data", os.environ["ENV_TYPE"], str(uuid.uuid1()) + ".pkl")

        df = get_historical_data(ticker, start, end, output_format='pandas')
        df.to_pickle(file_loc)

        path_dict[ticker] = file_loc
        if debug: print(f"Got {ticker} info from local store")

        save_data_info(path_dict)
        return df

    
    # have to actually read from file first to find whats missing
    # have to deal with date ranges
    file_loc = path_dict[ticker]
    df = pd.read_pickle(file_loc)
    local_start, local_end = min(df.index), max(df.index)
    one_day = pd.Timedelta("1D")
    temp_name = []
    if debug: print(f"local_start: {local_start}\tlocal_end: {local_end}")
    if end <= local_start:
        # gap on the left, pull from (start, local_start - 1D)
        temp_name.append((start, local_start - one_day))
    elif start >= local_end:
        # just get the data specified (local_end + 1D, end)
        temp_name.append((local_end + one_day, end))

    elif start >= local_start and end <= local_end:
        # in the middle, get it from local dataframe
        return df[start : end - one_day]

    elif start < local_start and end <= local_end:
        # grab from (start, local_start - 1D)
        temp_name.append((start, local_start - one_day))

    elif start < local_start and end > local_end:
        # both outside, get (start, local_start - 1D) and (local_end + 1D, end)
        temp_name.append((start, local_start - one_day))
        temp_name.append((local_end + one_day, end))

    elif start <= local_end and end > local_end:
        # the right side is peaking out, get (local_end + 1D, end)
        temp_name.append((local_end + one_day, end))

    while temp_name:
        start_, end_ = temp_name.pop(0)
        if debug: print(f"Getting {ticker} from IEX with start {start_} and end {end_}")
        df_ = get_historical_data(ticker, start_, end_, output_format="pandas")
        df = pd.concat([df, df_], axis=0)

    df = df.sort_index()
    df.to_pickle(file_loc)
    return df[start : end - one_day]  # remove all the + one_day and put them in df_ line

def get_sma(
    ticker:str, 
    interval:str, 
    time_periods: int, 
    timestamp: pd.Timestamp=None):
    from alpha_vantage.techindicators import TechIndicators
    import time

    # pkl path to the cache dataframe
    path = os.path.join(os.path.dirname(__file__), "data", os.environ["ENV_TYPE"], f"{ticker}_{interval}_sma{time_periods}.csv")
    if os.path.exists(path):
        # can just pull and return the saved object
        # print("Used local cache for SMA")
        df = pd.read_csv(path, index_col="date")
        df.index = pd.DatetimeIndex(df.index)
        return df

    """ None timestamp=None just means get the most recent"""
    sma = TechIndicators(key=ALPHA_VANTAGE, output_format="pandas").get_sma(
        ticker, 
        interval=interval, 
        time_period=time_periods,
    )
    
    if not sma:
        raise ValueError("Didn't manage to get SMA from alpha_vantage API")
    
    # save the csv
    df = pd.DataFrame(sma[0])    # apparently sma[0] _could_ be a string
    df.index = pd.DatetimeIndex(df.index)
    df.to_csv(path, index=True)

    # return the df
    print("Pulled SMA from API")
    return df

def timestamp_to_date(timestamp):
    ts = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    print(datetime.fromtimestamp(timestamp))
    print(f"turned timestamp from {timestamp} to {ts}")
    return ts