import pytest
import unittest

import pandas as pd
import numpy as np

from titan.portfolio import Portfolio
from canopus.secrets import set_test_env

set_test_env()

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(cash=100_000)
        self.portfolio.order("GOOG", 1, timestamp=pd.Timestamp('2020-02-03'))
        self.portfolio.order("AAPL", 10, timestamp=pd.Timestamp('2020-02-03'))
        self.portfolio.order("FB", 1, timestamp=pd.Timestamp('2020-02-03'))


    def test_position_vector(self):
        pos = self.portfolio.position_vector()        
        print(pos)

    def test_var(self):
        var = self.portfolio.VaR(historical=False)
        hist_var = self.portfolio.VaR(historical=True)
        print(f"var: {var}\thist_var: {hist_var}")

        # when running line 231, getting an erro when pulling price data at most rececnt date
        
    # def test_inc_var(self):
    #     with pytest.raises(NotImplementedError):
    #         self.portfolio.incremental_VaR()
