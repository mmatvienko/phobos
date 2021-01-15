from time import time
from titan.security import Security
from titan.portfolio import Portfolio
from pegasus.fe import diff, lag

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
        self.short_days = 50
        self.long_days = 200

        self.portfolio = portfolio
        self.run_name = run_name
        self.tickers = tickers
        self.results = pd.DataFrame(index=pd.DatetimeIndex([], name="time"), columns=[f"sma{self.short_days}", f"sma{self.long_days}", "price", "pv"])
        self.shorts = pd.Series([], index=pd.DatetimeIndex([]))
        self.longs = pd.Series([], index=pd.DatetimeIndex([]))
        self.fe = diff.Diff(1)

    def run(self, timestamp: pd.Timestamp=None):
        """  Given some time, run the strategy at that time.
        if timestamp is None, run live

        """
        for tick in self.tickers:
            sec = Security(tick)
            short_sma = sec.get_sma(
                self.short_days, 
                timestamp=timestamp,
            )
            long_sma = sec.get_sma(
                self.long_days, 
                timestamp=timestamp,
            )            
            price = sec.get_price(timestamp=timestamp)
            short_sma_slope = sec.get_sma_slope(
                self.short_days, 
                timestamp=timestamp
            )

            logging.info(f"SMA{self.short_days} = {short_sma.item()}\tSMA{self.long_days} = {long_sma.item()}")

            # if sma 50 < sma 200, sell
            if short_sma > long_sma:
                # go short
                # if self.portfolio.close_pos(tick, timestamp=timestamp):
                amt_owned = self.portfolio[tick]
                if self.portfolio.order(tick, -int(amt_owned*.66), timestamp=timestamp):
                    self.shorts[timestamp] = price

            elif short_sma < long_sma and short_sma_slope > 0:
                # go long
                cash = self.portfolio.cash
                amt = int(cash*0.5/price)   # TODO put this in the broker and see how we handle getting a different amount returned

                if self.portfolio.order(tick, amt, timestamp=timestamp):
                    self.longs[timestamp] = price

            price = sec.get_price(timestamp=timestamp)                
            row = pd.Series({
                f"sma{self.short_days}": short_sma,
                f"sma{self.long_days}": long_sma,
                "price": price,
                "diff": self.fe.step(price),
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

        colors = ["red", "blue", "orange", "green", "purple"]
        # do plotting stuff
        for col in df.columns:
            if col == "pv":
                ax2.plot(df[col], color=colors.pop(0), label=col)
            else:
                ax.plot(df[col], color=colors.pop(0), label=col)
        
        # show the purchase locations
        ax.plot(self.longs, 'g^')
        ax.plot(self.shorts, 'rv')
        ax.legend(loc=0)
        ax2.legend(loc=1)
        # show the plot
        plt.show()


        # could also include some post run computations
        return df

    def backfill(self, timestamp=None):
        """ Backfill all needed data for running step at timestamp
        """
        return None