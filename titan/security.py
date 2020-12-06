from datetime import datetime, date, timedelta
from canopus import utils
from iexfinance.stocks import Stock

import pandas as pd
import time

class Security():
    """ Maybe make a static class"""
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_source = None

    def get_price(self, timestamp=None):
        """ Gets the price for a security
        ticker : str - ticker of the security
        timestamp[optional] :   pd.Timestamp or datetime at which data should be pulled
                                probably used in backtesting, while None will be used in live
        """
        if not timestamp:
            # getting the current live price
            return Stock(self.ticker).get_price()

        ret = None
        # have to add one day since the look up is none inclusive.
        
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(utils.timestamp_to_date(timestamp))

        tmp_date = timestamp + timedelta(days=1)
        # TODO consider moving this while loop into whatever is calling `get_price`
        while ret is None:   # incase today isn't a trading day
            ret = self.history(end=tmp_date, time_frame="1D", price_type="close")
            tmp_date -= timedelta(days=1)

        return ret.item()

    def history(
        self, 
        start=None, 
        end=date.today() + timedelta(days=1), 
        time_frame="1Y", 
        price_type=None
        ):
        """
        For now only get the closing price
        start and end are datetime objects
        maybe change end=TODAY+1

        price type: can be something like close or open
        """
        
        if not start:
            start = end - pd.Timedelta(time_frame)

        df = utils.pull_historical_data(self.ticker, start, end)

        if df.empty:
            # not a trading day
            # seems to be getting called twice on 4th of july
            # print(f"returned none from position.history on {end} for {self.ticker}")
            return None
        
        # return the full row if no price type is specified
        if not price_type:
            return df
        
        return df[price_type]

    def get_sma(self, interval, time_periods, timestamp:pd.Timestamp=None):
        """
        interval: the amount of time between each data point
        time_periods: number of data points used to calculated the SMA
        """
        # go to the database here
        sma_frame = utils.get_sma(self.ticker, interval, time_periods, timestamp=timestamp)
        sma = sma_frame.loc[timestamp].item()
        return sma