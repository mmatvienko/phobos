from datetime import timedelta, date
from .models.auto_regressive import ARIMA

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
        pass
    
    def step_portfolio(self, date):
        # while running live maybe use Stock.get_open_close()
        for ticker in self.portfolio:
            # run the step for each ticker in the portfolio
            price_pred, curr_price = self.step_position(date, ticker)
            if price_pred > curr_price:
                # buy shit
                pass
            elif price_pred < curr_price:
                # sell shit
                pass
            
        self.portfolio.evaluate()
        
    def step_position(self, date, ticker):
        model = self.models[ticker]
        position = self.portfolio[ticker]
        datum = position.history(start=date, end=date+timedelta(days=1))
        prediction, stderr, conf_intrvl = model.step(datum)

        return prediction, position.get_price()
        
