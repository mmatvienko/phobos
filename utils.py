from .secrets import *
import os

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
