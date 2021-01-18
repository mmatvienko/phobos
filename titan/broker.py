from datetime import date
from abc import ABC

class Broker(ABC):
    """ Puts in a request with brokerage, waits on response.
    Based on results, the portfolio is updated.

    if date is None, we do a live transaction
    otherwise, we just return the price at the time provided
    """

    def __init__(self):
        self.broker_name = "base"

    def order(self, ticker, amt, price, date=None):
        raise NotImplementedError()
