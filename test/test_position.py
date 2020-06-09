import unittest
import datetime

import pandas as pd

from ..portfolio import Portfolio
from ..position import Position
from ..utils import set_test_env

set_test_env()

class TestGetters(unittest.TestCase):
    def test_get_returns(self):
        pos = Position("GOOG", 1, -1)
        start = datetime.date(year=2019, month=1, day=1)
        end = datetime.date(year=2020, month=1, day=1)

        history = pos.history(start=start, end=end)
        print(history)
        returns = pos.get_returns(start=start, end=end)
        print(returns)

class TestARIMA(unittest.TestCase):
    def test_arima(self):
        pos = Position("MSFT", 1, -1)
        pos.arima(1,1,1)
    
