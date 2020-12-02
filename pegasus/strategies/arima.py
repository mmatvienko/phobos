from datetime import timedelta, date, datetime
from titan.models.auto_regressive import ARIMA
import pandas as pd # temp_name

class ARIMAStrategy():
    """
    Should be backtestable
    given a portfolio will perform trades
    not sure if automatically, or if it will generate some sort of trade() object
    """
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.models = {}

        for ticker in self.portfolio:
            self.models[ticker] = ARIMA(5,1,1)


    def step(self, date=date.today()):
        """ step_portfolio only steps the models. while this step
        will run that, will run acquisition based on results??
        currently assume that this is run at 16:30 i.e. market close
        """
        self.step_portfolio(date)
    
    def step_portfolio(self, date):
        # NOTE: while running live maybe use Stock.get_open_close()
        self.portfolio.set_date(date)
        
        for ticker in self.portfolio:
            # run the step for each ticker in the portfolio
            price_pred, curr_price = self.step_position(date, ticker)
            if price_pred is None or curr_price is None:
                # the model is not ready to run yet, or we dont have data
                continue

            const = 10000//curr_price
            
            if price_pred > curr_price:
                # buy shit
                amt = self.portfolio.open_pos(ticker, amt=const)
                

            elif price_pred < curr_price:
                # sell shit
                self.portfolio.close_pos(ticker, amt=2*const)

        print(f" value at the end of {date}: ", self.portfolio.evaluate(date))

        for ticker in self.portfolio:
            pos = self.portfolio[ticker]
            print(f"\t{pos.get_amount()} of {ticker} @ {pos.get_price(date)}")

    def step_position(self, date, ticker):
        model = self.models[ticker]
        position = self.portfolio[ticker]
            
        datum = position.history(start=date, end=date+timedelta(days=1))
        
        if datum is None:
            # a holiday or just no data today
            return None, None

        tomorrow_pred, stderr, conf_intrvl = model.step(datum)

        # model wasn't ready to predict
        if tomorrow_pred is None:
            return None, None
        
        return tomorrow_pred, datum["close"].item()
        
