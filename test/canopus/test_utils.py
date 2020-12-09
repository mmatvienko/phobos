import unittest
import datetime, logging, sys
import pytest

import pandas as pd
import psycopg2 as psql

from canopus.utils import (
    pull_historical_data, 
    _check_table_health,
    timestamp_to_date,
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

    def test_pull_sql(self):
        import psycopg2 as psql

        if not os.environ["ENV_TYPE"]:
            raise ValueError("ENV_TYPE environment variable is not set, but is needed to run")
        
        db_name = os.environ["ENV_TYPE"] + "_db"
        con = psql.connect(
            host="localhost", 
            user="marcmatvienko",
            database=db_name, 
            password=None, 
        )
        print(f"connection established to {db_name}")
        res = pull_data_sql(
            con=con,
            table="SPY", 
            start=pd.Timestamp("2020-01-01"), 
            end=pd.Timestamp("2020-02-02"),
        )
        print(res)
        
    def test_convert(self):
        ts = pd.Timestamp("2020-02-01")
        recon = timestamp_to_date(ts.value/1e9)
        assert ts == pd.Timestamp(recon)

    def test_errors(self):
        with pytest.raises(ValueError):
            pull_historical_data("GOOG", "2019-01-01", pd.Timestamp("2019-01-01"))
        with pytest.raises(ValueError):
            pull_historical_data("GOOG", pd.Timestamp("2019-01-01"), "2019-01-01")

