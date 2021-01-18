from datetime import datetime, date, timedelta
from canopus import utils
from iexfinance.stocks import Stock

import pandas as pd
import time, os, logging
import psycopg2 as psql

class Security():
    """ Maybe make a static class"""
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_source = None
        self.con = psql.connect(
            host="localhost", 
            user="marcmatvienko",
            database=os.environ["ENV_TYPE"]+"_db", 
            password=None, 
        )
        self.con.autocommit = True # issue at _check_table_health, 
                                   # transaction started but not committed

        try:
            # TODO check if exists rather than try except.
            with self.con.cursor() as cur:
                cur.execute(f"CREATE TABLE {self.ticker} (time TIMESTAMP NOT NULL, PRIMARY KEY(time));")        
            self.con.commit()
        except:
            pass
            # TODO REMOVE logging.info("Didn't create new table, exists.")
        
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
            ret = self.history(start=timestamp, time_frame="1D", price_type="close")
            tmp_date -= timedelta(days=1)

        try:
            ret = ret.item()
        except:
            ret = ret.iloc[0].item()
            logging.warning(f"get_price for {self.ticker} got a date range from Security.history... Investigate this!")
            import ipdb; ipdb.set_trace()
        return ret

    def history(
        self, 
        start=date.today(), 
        end=None, 
        time_frame="1Y", 
        price_type=None
        ):
        """
        For now only get the closing price
        start and end are datetime objects
        maybe change end=TODAY+1

        price type: can be something like close or open
        """
        
        if not end:
            end = start   # getting data from the same day

        df = utils.pull_data_sql(
            con=self.con,
            table=self.ticker,
            start=start,
            end=end,
            cols=[price_type],
        )

        if df.empty:
            # not a trading day
            # seems to be getting called twice on 4th of july
            # print(f"returned none from position.history on {end} for {self.ticker}")
            return None
        
        # return the full row if no price type is specified
        if not price_type:
            return df
        
        return df[price_type]

    def get_sma_slope(self, time_periods, interval="daily", timestamp:pd.Timestamp=None):
        from pandas.tseries.offsets import BDay
        import trading_calendars as tc
        xnys = tc.get_calendar("XNYS")
        p = xnys.previous_open(timestamp)


        sma_frame = utils.pull_data_sql(
            self.con,
            self.ticker, 
            start=pd.Timestamp(str(p)[:10]),
            end=timestamp,
            cols=[f"sma{time_periods}"]
        )
        if len(sma_frame) == 1:
            # don't have enough data at his point, consider rolling back the date
            return 0

        frame = sma_frame
        sma_now = frame.iloc[1, 0]
        sma_prev = frame.iloc[0, 0]
        return sma_now - sma_prev

    def get_sma(self, time_periods, interval="daily", timestamp:pd.Timestamp=None):
        """
        interval: the amount of time between each data point (DAILY)
        time_periods: number of data points used to calculated the SMA
        """

        if timestamp is None:
            timestamp = pd.Timestamp.today()

        # go to the database here
        sma_frame = utils.pull_data_sql(
            self.con,
            self.ticker, 
            start=timestamp,
            end=timestamp,
            cols=[f"sma{time_periods}"]
        )
        # import ipdb; ipdb.set_trace()
        return sma_frame[timestamp : timestamp].iloc[0, 0]