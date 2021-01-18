import logging, sys, os, time
import trading_calendars as tc
import alpaca_trade_api as tradeapi


from datetime import datetime
from pegasus import strategies
from canopus.secrets import set_test_env, set_prod_env
from canopus import utils
from titan import portfolio, alpaca_broker

def run():
    # check for cached portfolio
    try:
        port = portfolio.Portfolio.read("alpaca_test")
    except:
        port = portfolio.Portfolio(cash=100_000, broker=alpaca_broker.AlpacaBroker())

    strategy = strategies.sma.SMA(portfolio)
    api = tradeapi.REST()

    if api.get_clock().is_open:
        strategy.run()
        port.write("alpaca")
    else:
        # doesn't run b/c no trading day
        # maybe checks for next trading dayt and runs
        # sees if any order are in place and modifies them
        pass


if __name__ == "__main__":
    set_test_env(data="alpaca")
    
    # setup logging logic
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.WARNING, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger('LOGGER_NAME')
    
    run()