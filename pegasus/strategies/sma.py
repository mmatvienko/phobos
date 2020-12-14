from time import time
from titan.security import Security
from titan.portfolio import Portfolio

import logging
import pandas as pd
import matplotlib.pyplot as plt

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
            if short_sma > long_sma:
                # go short
                order_res = self.portfolio.close_pos(tick, timestamp=timestamp)

            elif short_sma < long_sma:
                # go long
                order_res = self.portfolio.order(tick, 10, timestamp=timestamp)

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
        df = self.results
        fig = plt.figure()
        plt.xticks(rotation=45)
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        colors = ["red", "blue", "orange", "green"]
        # do plotting stuff
        for col in df.columns:
            if col == "pv":
                ax2.plot(df[col], color=colors.pop(0), label=col)
                # df[col].plot(secondary_y=True, legend=True)
            else:
                ax.plot(df[col], color=colors.pop(0), label=col)
                # df[col].plot(legend=True)
        
        ax.legend(loc=0)
        ax2.legend(loc=0)
        plt.show()
        # could also include some post run computations
        return df

    def backfill(self, timestamp=None):
        """ Backfill all needed data for running step at timestamp
        """
        return None