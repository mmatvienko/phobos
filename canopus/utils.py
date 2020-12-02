from canopus.secrets import *
from datetime import date, timedelta, datetime

import pandas as pd
import os
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

def get_sma(ticker, interval, time_periods, timestamp=None):
    from alpha_vantage.techindicators import TechIndicators
    import time

    """ None timestamp just means get the most recent"""
    sma = TechIndicators(key=ALPHA_VANTAGE, output_format="pandas").get_sma(
        ticker, 
        interval=interval, 
        time_period=time_periods,
    )
    
    if not sma:
        raise ValueError("Didn't manage to get SMA from alpha_vantage API")

    if timestamp and isinstance(timestamp, pd.Timestamp):
        date_index = timestamp_to_date(timestamp)
    else:
        date_index = sma[0].index[-1]

    return sma[0].loc[date_index]['SMA']

def timestamp_to_date(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%d-%m")