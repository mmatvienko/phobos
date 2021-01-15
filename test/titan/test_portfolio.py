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

    def test_insufficient_cash(self):

        self.portfolio.order("GE", 100_000, timestamp=pd.Timestamp('2020-02-03'))
        

        assert self.portfolio["GE"] == 0

    def test_position_vector(self):
        pos = self.portfolio.position_vector()        
        np.testing.assert_array_equal(pos[0], np.array([1, 10, 1]))

    def test_var(self):
        var = self.portfolio.VaR(historical=False, end=pd.Timestamp('2020-02-03'))
        hist_var = self.portfolio.VaR(historical=True, samples=5_000_000, end=pd.Timestamp('2020-02-03'))
        print(f"var: {var}\thist_var: {hist_var}")

    # TODO   
    # def test_inc_var(self):
    #     with pytest.raises(NotImplementedError):
    #         self.portfolio.incremental_VaR()


class TestSharpe(unittest.TestCase):
    def test_optimal(self):
        self.portfolio = Portfolio(cash=100_000)
        self.portfolio.order("GOOG", 1, timestamp=pd.Timestamp('2020-02-03'))
        self.portfolio.order("BYND", 1, timestamp=pd.Timestamp('2020-02-03'))
        self.portfolio.order("GE", 10, timestamp=pd.Timestamp('2020-02-03'))
        so, var_ = self.portfolio.sharpe_optimal(timestamp=pd.Timestamp('2020-12-14'))
        
        # TODO write out into function
        x = self.portfolio.weight_vector(timestamp=pd.Timestamp('2020-12-14'))
        pv = self.portfolio.price_vector(timestamp=pd.Timestamp('2020-12-14'))
        evalu = self.portfolio.evaluate(timestamp=pd.Timestamp('2020-12-14'))
        equ = (evalu-self.portfolio.cash)
        mov = ((so-x)*equ)/pv
        print(mov)
        self.portfolio.order("GOOG", -0.25981319, timestamp=pd.Timestamp('2020-12-14'))
        self.portfolio.order("BYND", 6.89360815, timestamp=pd.Timestamp('2020-12-14'))
        self.portfolio.order("GE", -45.94384114, timestamp=pd.Timestamp('2020-12-14'))
        so, _ = self.portfolio.sharpe_optimal(timestamp=pd.Timestamp('2020-12-14'))
        curr_weight = self.portfolio.weight_vector(timestamp=pd.Timestamp('2020-12-14'))
        
        # checking that correction is sufficient to make optimal
        np.testing.assert_array_almost_equal(so, curr_weight[0])
        import ipdb; ipdb.set_trace()
