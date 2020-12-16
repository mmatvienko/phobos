import unittest
import datetime, logging, sys, os
import pytest

import pandas as pd
import psycopg2 as psql

from canopus.utils import (
    # pull_historical_data, 
    _check_table_health,
    timestamp_to_date,
    pull_data_sql,
    get_missing_dates,
    _pull_col_data,
    )
from canopus.secrets import set_test_env

set_test_env()

stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[stdout_handler]
)

class TestTableHealth(unittest.TestCase):
    def setUp(self):
        self.con = psql.connect(
            host="localhost", 
            user="marcmatvienko",
            database="testing", 
            password=None, 
        )        
    def test_no_table(self):
        # fails properly when table doesnt exist
        # returns the right columns
        cur = self.con.cursor()
        cur.execute("DROP TABLE IF EXISTS spy")    
        cols = _check_table_health(self.con, "spy", ["price"])
        assert cols[0] == "price"

    def test_missing_columns(self):
        # returns the correct missing columns and table now contains them
        cur = self.con.cursor()
        cur.execute("DROP TABLE IF EXISTS spy")    
        cur.execute("CREATE TABLE spy (time TIMESTAMP, price float)")        
        cols = _check_table_health(self.con, "spy", ["price", "sma"])
        assert ("sma" in cols)


class TestGetMissingDates(unittest.TestCase):
    def setUp(self):
        self.db_left = pd.Timestamp("2020-01-01")
        self.db_right = pd.Timestamp("2020-02-01")

    def test_left_out(self):
        left = pd.Timestamp("2019-01-01")
        right = pd.Timestamp("2020-01-15")
        dates = get_missing_dates(self.db_left, self.db_right, left, right)
        assert (
            dates[0][0] == left and 
            dates[0][1] == self.db_left - pd.Timedelta("1D")
            )
    def test_right_out(self):
        left = pd.Timestamp("2020-01-15")
        right = pd.Timestamp("2021-01-01")
        dates = get_missing_dates(self.db_left, self.db_right, left, right)
        assert (
            dates[0][0] == self.db_right + pd.Timedelta("1D") and 
            dates[0][1] == right
            )

    def test_both_out(self):
        left = pd.Timestamp("2019-01-01")
        right = pd.Timestamp("2021-01-01")
        dates = get_missing_dates(self.db_left, self.db_right, left, right)
        assert (
            dates[0][0] == left and 
            dates[0][1] == self.db_left - pd.Timedelta("1D") and 
            dates[1][0] == self.db_right + pd.Timedelta("1D") and
            dates[1][1] == right
            )

    def test_both_in(self):
        left = pd.Timestamp("2020-01-05")
        right = pd.Timestamp("2020-01-15")
        dates = get_missing_dates(self.db_left, self.db_right, left, right)
        assert dates is None

    def test_single_boundary_right(self):
        dates = get_missing_dates(
            pd.Timestamp('2019-01-02 00:00:00', freq='D'),
            pd.Timestamp('2019-01-28 00:00:00', freq='D'),
            pd.Timestamp('2019-01-28 00:00:00', freq='D'),
            pd.Timestamp('2019-01-28 00:00:00', freq='D'),
            )
        assert dates is None

    def test_single_boundary_left(self):
        dates = get_missing_dates(
            pd.Timestamp('2019-01-02 00:00:00', freq='D'),
            pd.Timestamp('2019-01-28 00:00:00', freq='D'),
            pd.Timestamp('2019-01-02 00:00:00', freq='D'),
            pd.Timestamp('2019-01-02 00:00:00', freq='D'),
            )
        assert dates is None

    def test_both_left(self):
        left = pd.Timestamp("2019-01-01")
        right = pd.Timestamp("2019-02-01")
        dates = get_missing_dates(self.db_left, self.db_right, left, right)
        assert (
            dates[0][0] == left and 
            dates[0][1] == self.db_left - pd.Timedelta("1D")
            )

    def test_both_right(self):
        left = pd.Timestamp("2021-01-01")
        right = pd.Timestamp("2021-02-01")
        dates = get_missing_dates(self.db_left, self.db_right, left, right)
        assert (
            dates[0][0] == self.db_right + pd.Timedelta("1D") and 
            dates[0][1] == right
            )

    def test_invalid_order(self):
        left = pd.Timestamp("2022-01-01")
        right = pd.Timestamp("2019-01-01")
        with pytest.raises(ValueError):
            get_missing_dates(self.db_left, self.db_right, left, right)


class TestPullColData(unittest.TestCase):
    """ check health should have run before this,
    so no need to do any column checks.

    WLOG will test for 'price' only
    """
    def setUp(self):
        self.con = psql.connect(
            host="localhost", 
            user="marcmatvienko",
            database="testing", 
            password=None, 
        ) 
    def test_pull_price(self):
        kwargs = {
            "start": pd.Timestamp("2020-01-02"), 
            "end": pd.Timestamp("2020-01-03"),
            }
        df = _pull_col_data(
            table="spy", 
            col="open",
            **kwargs,
        )

        index = pd.date_range(
            start=pd.Timestamp("2020-01-02"),
            periods=2,
            freq="1D"
        )
        # can't really compare prices since the testing API is random
        assert all(index == df.index)

    def test_pull_sma(self):
        df = _pull_col_data(
            table="spy",
            col="sma50",
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-02-02"),            
        )
        expected = {0: 95.7273, 234:94.0212, 3333: 132.4932}
        for loc in expected:
            assert df.iloc[loc] == expected[loc]


class TestPullDataSql(unittest.TestCase):
    def setUp(self):
        self.con = psql.connect(
            host="localhost", 
            user="marcmatvienko",
            database="testing", 
            password=None, 
        ) 
        print(f"connection established to testing db")
        
    def test_pull_single_col(self):
        cur = self.con.cursor()
        cur.execute("DROP TABLE IF EXISTS spy")    
        cur.execute("CREATE TABLE spy (time TIMESTAMP NOT NULL, PRIMARY KEY(time));")        

        res1 = pull_data_sql(
            con=self.con,
            table="SPY", 
            start=pd.Timestamp("2020-01-01"), 
            end=pd.Timestamp("2020-02-01"),
            cols=["open"],
        )

        # make sure this one doesn't call iex
        res2 = pull_data_sql(
            con=self.con,
            table="SPY", 
            start=pd.Timestamp("2020-01-01"), 
            end=pd.Timestamp("2020-02-01"),
            cols=["open"],
        )
        # wouldnt be different since iex API gives different results every time
        assert all(res1 == res2)

    def test_pull_multi_col(self):
        res = pull_data_sql(
            con=self.con,
            table="SPY", 
            start=pd.Timestamp("2020-02-01"), 
            end=pd.Timestamp("2020-02-03"),
            cols=["open", "close"],
        )
        assert res.shape == (1,2)
        assert isinstance(res["open"].item(), float)