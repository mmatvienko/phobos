from titan.broker import Broker
from titan.security import Security

import psycopg2 as psql

import os


class DumbBroker(Broker):
    def __init__(self):
        self.broker_name = "dumb"
        self.cash = 100_000
        self.con = None

    def open(self, ticker, amt, price=None, timestamp=None):
        """ open a position. whether it be buy to open or sell
        amt - specifies short or long
        price - if not specified we have market order, else limit
        """
        price = Security(ticker).get_price(timestamp=timestamp)
        return  (amt, price)

    def close(self, ticker, amt, price=None, timestamp=None):
        price = Security(ticker).get_price(timestamp=timestamp)
        return  (amt, price)

    # def set_con(self):
    #     """ Call once os.environ['ENV_TYPE'] has been set """
    #     if "ENV_TYPE" in os.environ:
    #         self.con = psql.connect(
    #             host="localhost", 
    #             user="marcmatvienko",
    #             database=os.environ["ENV_TYPE"]+"_db", 
    #             password=None, 
    #         )
    #     else:
    #         raise ValueError("ENV_TYPE environment variable has to be set.")
