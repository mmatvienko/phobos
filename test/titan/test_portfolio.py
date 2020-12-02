import pytest
import unittest

import pandas as pd
import numpy as np

from titan.portfolio import Portfolio
from titan.position import Position
from canopus.utils import set_test_env

set_test_env()

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio()
        self.portfolio.open_pos("GOOG", 1, timestamp=1606871227)
        self.portfolio.open_pos("GE", 10, timestamp=1606871227)
        self.portfolio.open_pos("SPY", 1, timestamp=1606871227)


    def test_position_vector(self):
        pos = self.portfolio.position_vector()        
        print(pos)
        assert all(pos[0] == [1, 10, 1])

    def test_var(self):
        var = self.portfolio.VaR(historical=False)
        hist_var = self.portfolio.VaR(historical=True)
        print(f"var: {var}\thist_var: {hist_var}")
        
    # def test_inc_var(self):
    #     with pytest.raises(NotImplementedError):
    #         self.portfolio.incremental_VaR()
