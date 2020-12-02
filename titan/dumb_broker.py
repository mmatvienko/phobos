from titan.broker import Broker
from titan.security import Security

class DumbBroker(Broker):
    def __init__(self):
        self.broker_name = "dumb"

    def open(self, ticker, amt, price=None, timestamp=None):
        """ open a position. whether it be buy to open or sell
        amt - specifies short or long
        price - if not specified we have market order, else limit
        """
        price = Security(ticker).get_price(timestamp=timestamp)
        return  (amt, price)

    def close(self, ticker, amt, price=None, timestamp=None):
        price = Security(ticker).get_price(timestamp=timestamp)
        return  (amt, price)        