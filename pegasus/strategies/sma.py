from time import time

class SMA():
    def __init__(self, portfolio, tickers=["SPY"]):
        """ SMA will buy and sell based on SMA 50 vs 200 cross
        Will do it for all tickers. Could use metaflow step on each tickers
        A strategy is the operation of some trades at a single tiem stamp
        The runner is what will handle running it at repeated periods
        """
        self.portfolio = portfolio

    def run(self, timestamp=None):
        for tick in tickers:
            # TODO
  
        return None

    def backfill(self, timestamp=None):
        """ Backfill all needed data for running step at timestamp
        """
        return None