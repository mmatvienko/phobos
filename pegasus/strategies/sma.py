from time import time
from titan.security import Security
from titan.portfolio import Portfolio

import logging
import pandas as pd

class SMA():
    def __init__(self, portfolio, tickers=["SPY"], run_name:str="sma"):
        """ SMA will buy and sell based on SMA 50 vs 200 cross
        Will do it for all tickers. Could use metaflow step on each tickers
        A strategy is the operation of some trades at a single tiem stamp
        The runner is what will handle running it at repeated periods
        """
        self.portfolio = portfolio
        self.run_name = run_name
        self.tickers = tickers
        self.results = pd.DataFrame(index=pd.DatetimeIndex([], name="time"), columns=["sma50", "sma200", "price", "pv"])

    def run(self, timestamp: pd.Timestamp=None):
        """  Given some time, run the strategy at that time.
        if timestamp is None, run live

        """
        for tick in self.tickers:
            sec = Security(tick)
            short_days = 50
            short_sma = sec.get_sma(
                short_days, 
                timestamp=timestamp,
                )
            long_days = 200
            long_sma = sec.get_sma(
                long_days, 
                timestamp=timestamp,
                )            
            logging.info(f"SMA{short_days} = {short_sma.item()}\tSMA{long_days} = {long_sma.item()}")
            # if sma 50 < sma 200, sell
            if short_sma < long_sma:
                # go short
                self.portfolio.close_pos(tick, timestamp=timestamp)

            elif short_sma > long_sma:
                # go long
                self.portfolio.order(tick, 10, timestamp=timestamp)

            row = pd.Series({
                "sma50": short_sma,
                "sma200": long_sma,
                "price": sec.get_price(timestamp=timestamp),
                "pv": self.portfolio.evaluate(timestamp=timestamp),
            }, name=timestamp)

            self.results = self.results.append(
                row,
                ignore_index=False,
            )
            

    def get_results(self):
        # do some sort of summary.
        # could also include some post run computations
        return self.results

    def backfill(self, timestamp=None):
        """ Backfill all needed data for running step at timestamp
        """
        return None