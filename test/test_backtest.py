import pytest
import unittest

import pandas as pd
import numpy as np

from datetime import date, timedelta
from titan.portfolio import Portfolio
from titan.position import Position
from canopus.utils import set_test_env, backtest_strategy
from titan.strategy import ARIMAStrategy
set_test_env()

class TestBacktest(unittest.TestCase):

    def test_backtest_strategy(self):
        portfolio = Portfolio(cash=100_000)
        tickers = ["MSFT", "GOOG"]

        for ticker in tickers:
            portfolio.add_position(Position(ticker))
            
        strategy = ARIMAStrategy(portfolio)
        
        end_date = date.today()
        start_date = end_date - pd.Timedelta("12W")

        backtest_strategy(strategy, start_date, end_date)
