from .secrets import *

import pandas as pd
import os
import json
import uuid

from iexfinance.stocks import get_historical_data

def set_test_env():
    os.environ["IEX_API_VERSION"] = iex_sandbox_endpoint
    os.environ["IEX_TOKEN"] = iex_sandbox_secret

    os.environ["APCA_API_BASE_URL"] = alpaca_endpoint
    os.environ["APCA_API_KEY_ID"] = alpaca_api_id
    os.environ["APCA_API_SECRET_KEY"] = alpaca_secret    

def set_prod_env():
    os.environ["IEX_API_VERSION"] = iex_endpoint
    os.environ["IEX_TOKEN"] = iex_secret

    os.environ["APCA_API_BASE_URL"] = alpaca_endpoint
    os.environ["APCA_API_KEY_ID"] = alpaca_api_id
    os.environ["APCA_API_SECRET_KEY"] = alpaca_secret        

def get_data_info():
    """
    Reads in a json file that is a mapping between tickers and their files names
    """

    path = "data/info.json"

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
    path = "data/info.json"
    with open(path, "w") as f:
        json.dump(path_dict, f)

def pull_historical_data(ticker, start, end):
    """ If there are gaps between local data and what the users need,
    we will fill in those gaps
    TODO:    needs extensive testing
    """

    path_dict = get_data_info()
    
    if ticker not in path_dict:
        # doesnt exist yet, can just read in the whole thing and save it
        file_loc = "data/" + str(uuid.uuid1()) + ".pkl"

        df = get_historical_data(ticker, start, end, output_format='pandas')
        df.to_pickle(file_loc)

        path_dict[ticker] = file_loc

        print(f"pulled data from cloud")
    
        save_data_info(path_dict)
        return df

    # have to actually read from file first to find whats missing
    # have to deal with date ranges
    file_loc = path_dict[ticker]
    df = pd.read_pickle(file_loc)
    local_start, local_end = min(df.index), max(df.index)
    one_day = pd.Timedelta("1D")
    temp_name = []

    if end <= local_start:
        # gap on the left, pull from (start, local_start - 1D)
        temp_name.append((start, local_start - one_day))
    elif start >= local_end:
        # just get the data specified (local_end + 1D, end)
        temp_name.append((local_end + one_day, end))

    elif start >= local_start and end <= local_end:
        # in the middle, get it from local dataframe
        return df[start : end]

    elif start < local_start and end < local_end:
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
        df_ = get_historical_data(ticker, start_, end_, output_format="pandas")
        df = pd.concat([df, df_], axis=0)

    df = df.sort_index()
    save_data_info(path_dict)   # i dont think there is anything to save. remove?
    return df[start : end]

def backtest_strategy(strategy, ticker, start_date, end_date):
    """
    takes a strat and backtests it
    not sure if it should be for a ticker or for portfolio
    """
    pass

def backtest_var(portfolio, start_date, end_date):
    """
    Takes portfolio. then just without strategy, sees what the
    VaR is for every day, plots it, and then also plots actual returns on top
    """
    pass
