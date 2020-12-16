import unittest, pytest
import datetime, time
import pandas as pd

from titan.portfolio import Portfolio
from titan.security import Security
from canopus.secrets import set_test_env, set_prod_env

set_test_env()
# set_prod_env()

class TestGetters(unittest.TestCase):
    def setUp(self):
        self.security = Security("SPY")
  
    def test_get_sma(self):
        # timestamp == Dec 1, 2020
        sma = self.security.get_sma(50, timestamp=pd.Timestamp('2020-12-01'))
        assert 3e2 < sma

        with pytest.raises(RuntimeError):
            self.security.get_sma(time_periods=50, timestamp=None)

    def test_get_price(self):
        price = self.security.get_price()   # will change and will be random when on test env

        price = self.security.get_price(timestamp=pd.Timestamp("2020-01-01"))
        assert price > 325.51   # saved in DB

    def test_history(self):
        history = self.security.history(
            start=pd.Timestamp("2020-12-01"), 
            time_frame="2W", 
            price_type='close',
        )
        expected_results = [376.82, 372.81, 365.02, 357.29, 360.0, 377.28, 376.01, 365.59, 366.99]
        assert all(expected_results == history)