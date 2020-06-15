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

    def step(self, datum):
        # how will this work with data for multiple positions?
        pass
