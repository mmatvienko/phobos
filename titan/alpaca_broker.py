import alpaca_trade_api as tradeapi
import time

from canopus.secrets import alpaca_endpoint, alpaca_api_id, alpaca_secret
from titan.broker import Broker

class AlpacaBroker(Broker):
    def __init__(self):
        self.broker_name = "alpaca_broker"
        self.api = tradeapi.REST(
            alpaca_api_id, 
            alpaca_secret, 
            base_url=alpaca_endpoint
            )
        account = self.api.get_account()
        print(account)
        print(self.api.list_positions())

    def get_api(self):
        return self.api

    def order(self, ticker, amt, price=None, timestamp=None):
        """ open a position. whether it be buy to open or sell
        amt - specifies short or long
        price - if not specified we have market order, else limit
        """
        if amt < 0:
            side = "sell"
        else:
            side = "buy"

        order = self.api.submit_order(
            symbol=ticker,
            qty=abs(amt),
            side=side,
            type='market',
            time_in_force='gtc',
        )
        order_result = None
        ATTEMPTS, i = 5, 0
        while (order_result is None or order_result.filled_at is None) and i < ATTEMPTS:
            order_result = self.api.get_order(order.id)
            time.sleep(0.01) # TODO open a socket and wait for response instead
            i += 1

        if order_result.filled_at is None:
            # failed order
            return (-1, -1)

        price = order_result.filled_avg_price
        amt = order_result.filled_qty
        return (amt, price)