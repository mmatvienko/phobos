import pytest
import unittest

import pandas as pd

from ..portfolio import Portfolio
from ..position import Position
from ..utils import set_test_env

set_test_env()

class TestVaR(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio()
        self.portfolio.add_position(Position("GOOG", 1, 1000.00))
        self.portfolio.add_position(Position("GE", 10, 29.00))
        self.portfolio.add_position(Position("SPY", 1, 300.00))

    def test_position_vector(self):
        pos = self.portfolio.position_vector()        
        assert all(pos[0] == [1, 10, 1])

    def test_var(self):
        var = self.portfolio.VaR(historical=False)

    def test_inc_var(self):
        with pytest.raises(NotImplementedError):
            self.portfolio.incremental_VaR()
