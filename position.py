import pandas as pd

from datetime import datetime
from iexfinance.stocks import get_historical_data, Stock

class Position():
    def __init__(self, ticker, amount, entry_price, entry_date=pd.to_datetime("today")):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.amount = amount
        # verify that the entry price is within opening and closing of the entry date
        
    def close(self):
        pass

    def open(self):
        pass

    def get_amount(self):
        return self.amount
    
    def get_price(self, time=None):
        if not time:
            return Stock(self.ticker).get_price()
    
    def history(self, start=None, end=pd.to_datetime("today"), time_frame="1Y", price_type="open"):
        """
        For now only get the closing price
        start and end are datetime objects
        """

        if not start:
            start = end - pd.Timedelta(time_frame)

        df = get_historical_data(self.ticker, start, end, output_format='pandas')
        return df[price_type]

    def get_returns(self, start, end, time_frame="1Y", price_type="open"):
        prices = self.history(start, end)
        historical_returns = []
        for i in range(len(prices) - 1):
            historical_returns.append((prices[i + 1] - prices[i]) / prices[i])
            
        return pd.DataFrame(historical_returns, columns=[self.ticker], index=prices.index[:-1])
