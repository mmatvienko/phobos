import logging, sys

import pandas as pd
import trading_calendars as tc

from pegasus.strategies.sma import SMA
from titan.portfolio import Portfolio
from canopus.secrets import set_test_env


def run():
    strat = SMA(Portfolio(cash=100_000))
    nasdaq = tc.get_calendar("NASDAQ")

    final_date = None
    for date in pd.date_range(start=pd.Timestamp('2018-01-01'), periods=365, freq="1D"):

        if not nasdaq.is_session(date):
            logging.info(f"Skipping {date}")
            continue
        logging.info(f'Running on {date}')

        strat.run(timestamp=date)
        final_date = date

    print(strat.portfolio.evaluate(timestamp=final_date))
    print(strat.portfolio.cash)
    strat.get_results()

if __name__ == "__main__":
    set_test_env()
    
    # setup logging logic
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.DEBUG, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger('LOGGER_NAME')
    
    run()