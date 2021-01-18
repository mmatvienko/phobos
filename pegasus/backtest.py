import logging, sys, os

import pandas as pd
import trading_calendars as tc
import psycopg2 as psql

from pegasus.strategies.sma import SMA
from titan.portfolio import Portfolio
from titan.security import Security
from canopus.secrets import set_test_env, set_prod_env
from canopus.utils import pull_data_sql

def run():
    """ TODO: split of training dates and do a k fold backtest
    average return across all windows

    """
    strat = SMA(Portfolio(cash=100_000))
    nasdaq = tc.get_calendar("NASDAQ")

    # pre pull price data
    con = psql.connect(
        host="localhost", 
        user="marcmatvienko",
        database=os.environ["ENV_TYPE"]+"_db", 
        password=None, 
    )
    start = nasdaq.next_open(pd.Timestamp('2019-01-02'))
    start = pd.Timestamp(str(start)[:10])
    end = nasdaq.previous_close(start + pd.Timedelta("400D"))
    end = pd.Timestamp(str(end)[:10])

    for tick in strat.tickers:
        pull_data_sql(
            con, 
            tick, 
            end,
            end,
            cols=["close"]
        )

    final_date = None
    for date in pd.date_range(start=start, end=end, freq="1D"):

        if not nasdaq.is_session(date):
            logging.info(f"Skipping {date}")
            continue
        logging.info(f'Running on {date}')

        strat.run(timestamp=date)
        final_date = date

    print("Portfolio value: ", strat.portfolio.evaluate(timestamp=final_date))
    print("Portfolio cash: ", strat.portfolio.cash)
    init_amt = 100_000 / Security("spy").get_price(timestamp=start)
    final_val = init_amt*Security("spy").get_price(timestamp=end)
    print("Buy and hold value: ", final_val)
    strat.get_results()

if __name__ == "__main__":
    set_test_env()
    
    # setup logging logic
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger('LOGGER_NAME')
    
    run()