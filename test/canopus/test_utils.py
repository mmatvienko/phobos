import unittest
import datetime
import pytest

import pandas as pd

from canopus.utils import set_test_env, pull_historical_data

set_test_env()

class TestPullHistorical(unittest.TestCase):
    def test_pull(self):
        ticker = "GOOG"
        start = pd.Timestamp("2019-01-01")
        end = pd.Timestamp("2019-02-01")

        df = pull_historical_data(ticker, start, end)
        assert len(df) == 21

        end_ = pd.Timestamp("2019-03-01")
        df = pull_historical_data(ticker, start, end_)
        assert len(df) == 40

        df = pull_historical_data(ticker,pd.Timestamp("2019-01-15"), end)
        assert len(df) == 12

    def test_errors(self):
        with pytest.raises(ValueError):
            pull_historical_data("GOOG", "2019-01-01", pd.Timestamp("2019-01-01"))
        with pytest.raises(ValueError):
            pull_historical_data("GOOG", pd.Timestamp("2019-01-01"), "2019-01-01")
