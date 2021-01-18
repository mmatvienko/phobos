from titan.broker import Broker
from titan.security import Security

import os


class DumbBroker(Broker):
    def __init__(self):
        self.broker_name = "dumb"
        self.cash = 100_000
        self.con = None

    def order(self, ticker, amt, price=None, timestamp=None):
        """ open a position. whether it be buy to open or sell
        amt - specifies short or long
        price - if not specified we have market order, else limit
        """
        price = Security(ticker).get_price(timestamp=timestamp)
        return  (amt, price)