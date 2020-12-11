from canopus.secrets import ALPHA_VANTAGE
from datetime import date, timedelta, datetime

import pandas as pd
import os, logging, time, json, uuid

from iexfinance.stocks import get_historical_data
from alpha_vantage.techindicators import TechIndicators


one_day = pd.Timedelta("1D")


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
        logging.info(f"Created missing table for {table}")
        missing_cols = cols
    else:
        missing_cols = []
        # check for price column
        columns = [x[3] for x in fetched]
        logging.info(columns)
        for col in cols:
            if col not in columns:
                # column doesn't exist in table
                query = f"ALTER TABLE {table} ADD {col} {col_types.get(col, 'FLOAT')};"
                cur.execute(query)
                missing_cols.append(col)
        logging.info(f"Adding missing columns: {', '.join(missing_cols)}")

    cur.close()
    return missing_cols

def _pull_col_data(table, col, start, end):
    """ Pulls data from appropriate source 
    in preperation to append to sql db

    parameters:
    col: name of the data needed to be pulled

    return:
    df containing the row of needed information and perhaps more
    """
    if col.lower() in ("open", "close", "low", "high", "volume"):
        try:
            start = start
            end = end
        except:
            raise KeyError(f"When pulling {col} data, you should include start and end times in kwargs")
        df = get_historical_data(table, start, end, output_format='pandas')[col]

    elif col.lower()[:3] == "sma":
        try:
            time_periods = int(col[3:])
            interval = "daily"
        except ValueError:
            raise ValueError(f"Name of column starting with 'sma' has to end with an int, not {col[3:]}")
        except KeyError:
            raise KeyError(f"When data {col}, kwargs should include 'interval'")
        
        sma = TechIndicators(key=ALPHA_VANTAGE, output_format="pandas").get_sma(
            table, 
            interval=interval,   # something like "daily"
            time_period=time_periods,  # would capture 20 in sma20
        )
        df = pd.DataFrame(sma[0])    # apparently sma[0] _could_ be a string
        df.index = pd.DatetimeIndex(df.index)
    else:
        raise ValueError(f"Pulling {col} data is not yet supported")

    df.index = df.index.rename("time")
    return df

def pull_data_sql(con, table, start, end, cols=["price"], debug=False):
    """ Try to make cols a list so you can pull multiple attr if needed
    
    parameters:
    con: the psycopg2 connection to the db
    table: most likely the ticker 
    start: time at which data should start
    end: time at which data should end
    cols: the different data we need (e.g. price, sma,)

    returns:
    dataframe
    """
    table = table.lower()
    cur = con.cursor()

    # missing_cols is the data you need to pull in full
    missing_cols = _check_table_health(con, table, cols)
    final_df = pd.DataFrame([], index=pd.DatetimeIndex([]))
    final_df.index.name = "time"

    for col in cols:
        print(f"Col: {col}")
        col_data = pd.DataFrame()

        # was not missing, some data probably exists
        if col not in missing_cols:
            # figure out what time range we already have
            query = f"SELECT MIN(time) FROM {table} WHERE {col} IS NOT NULL;"
            cur.execute(query)
            db_start = cur.fetchone()

            query = f"SELECT MAX(time) FROM {table} WHERE {col} IS NOT NULL;"
            cur.execute(query)
            db_end = cur.fetchone()
            
            if db_start is None or db_end is None:
                raise ValueError(f"There doesn't seem to be any data in {col}, or the query didn't work.")

            missing_dates = get_missing_dates(db_start, db_end, start, end)
        # all missing and can pull full date range
        else:
            missing_dates = None

        while missing_dates:
                # pull missing data from source
                start_, end_ = missing_dates.pop(0)
                pulled_data = _pull_col_data(table, col, start_, end_)
                # pulled_data.index = pulled_data.index.rename("time")

                # save data to sql and 
                # append it to column that will part of returned df
                series_to_sql(pulled_data, con, table)
                # pulled_data.to_sql(table, con)
                col_data.append(pulled_data)

        col_data = _pull_col_data(table, col, start, end)
        series_to_sql(col_data, con, table)

        final_df = final_df.merge(
            col_data, how='outer', on='time'
        )
        con.commit()
    
    return final_df


def _build_query(values, table: str, col: str):
    """ build query """
    if not (isinstance(table, str) and isinstance(col, str)):
        raise ValueError(f"table: {table} or col: {col} aren't strings.")

    query = f"INSERT INTO {table} (time, {col}) VALUES "
    val_list = []
    for idx, val in values.items():
        # for different types, decide to have quotes or not
        # like '{idx}' -vs- {val}
        val_list.append(f"('{idx}', {val})")

    query += ",".join(val_list)
    query += ';'
    
    return query


def series_to_sql(s: pd.Series, con, table, chunk_size=10):
    """ Goal is for the series to be updated to the db """
    cur = con.cursor()

    for i in range(0, len(s), chunk_size):
        query = _build_query(
            s.iloc[i: i + chunk_size],
            table,
            s.name,
        )
        print(f"exectuing:\n {query}")
        cur.execute(query)

# def pull_price(ticker, start, end):
#     # do stuff needed to pull the price
#     return pd.DataFrame()

# pull_func = {"price": pull_price}

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
    temp_name = get_missing_dates(local_start, local_end, start, end)
    isnt_none = temp_name is not None

    # should be skipped if temp_name is none, and go straight to return
    while temp_name:
        start_, end_ = temp_name.pop(0)
        if debug: print(f"Getting {ticker} from IEX with start {start_} and end {end_}")
        df_ = get_historical_data(ticker, start_, end_, output_format="pandas")
        df = pd.concat([df, df_], axis=0)
    
    # only sort and pickle if we actually added some data
    if isnt_none:
        df = df.sort_index()
        df.to_pickle(file_loc)

    return df[start : end - one_day]  # remove all the + one_day and put them in df_ line



def get_missing_dates(db_start, db_end, start, end, debug=False):
    if start > end:
        raise ValueError(f"The start date {start} cannot be after the end date {end}.")

    missing_dates = []
    if debug: print(f"db_start: {db_start}\tdb_end: {db_end}")
    
    if end <= db_start:
        # gap on the left, pull from (start, db_start - 1D)
        missing_dates.append((start, db_start - one_day))
    elif start >= db_end:
        # just get the data specified (db_end + 1D, end)
        missing_dates.append((db_end + one_day, end))

    elif start >= db_start and end <= db_end:
        # in the middle, get it from local dataframe
        return None

    elif start < db_start and end <= db_end:
        # grab from (start, db_start - 1D)
        missing_dates.append((start, db_start - one_day))

    elif start < db_start and end > db_end:
        # both outside, get (start, db_start - 1D) and (db_end + 1D, end)
        missing_dates.append((start, db_start - one_day))
        missing_dates.append((db_end + one_day, end))

    elif start <= db_end and end > db_end:
        # the right side is peaking out, get (db_end + 1D, end)
        missing_dates.append((db_end + one_day, end))

    return missing_dates

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