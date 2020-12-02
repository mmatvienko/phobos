import pandas as pd
import numpy as np

from canopus.utils import pull_historical_data
from datetime import datetime, date, timedelta
from iexfinance.stocks import Stock

class Position():
    def __init__(self, ticker):
        self.ticker = ticker
        self.entry_date = None
        self.avg_cost = 0
        self.amount = 0
        self.date = None        # date will be incremented in backtesting.

        # TODO verify that the entry price is within opening and closing of the entry date

    def open(self, amt, available_funds):
        curr_price = self.get_price()

        if curr_price is None:
            # today is not a trading day, can't open
            return None
        
        amt = min(available_funds//curr_price, amt)  # incase funds aren't enough to fullfil the order
        self.avg_cost = (self.avg_cost * self.amount + curr_price * amt) / (self.amount + amt)
        
        print(f"amt: {amt}")
        # only set the entry date if the position has not been entered yet
        if not self.amount:
            self.entry_date = self.date

        self.amount += amt

        # return total amount of money spent
        return curr_price * amt
    
    def close(self, amt):
        self.amount -= amt

        # return the amount of money received for sale
        return amt*self.get_price()
        
    def get_amount(self):
        return self.amount

    def get_value(self, date=date.today()):
        return self.get_amount() * self.get_price()
    
    def get_price(self, live=False):
        """
        Returns none if not a trading day
        """
        if live:
            return Stock(self.ticker).get_price()
        else:
            ret = None
            tmp_date = self.date + timedelta(days=1)
            while ret is None:
                ret = self.history(end=tmp_date, time_frame="1D", price_type="close")
                tmp_date -= timedelta(days=1)

            return ret.item()
        
    def history(self, start=None, end=date.today() + timedelta(days=1), time_frame="1Y", price_type=None):
        """
        For now only get the closing price
        start and end are datetime objects
        maybe change end=TODAY+1
        """
        
        if not start:
            start = end - pd.Timedelta(time_frame)

        df = pull_historical_data(self.ticker, start, end)

        if df.empty:
            # not a trading day
            # seems to be getting called twice on 4th of july
            # print(f"returned none from position.history on {end} for {self.ticker}")
            return None
        
        # return the full row if no price type is specified
        if not price_type:
            return df
        
        return df[price_type]

    def get_returns(self, start, end, time_frame="1Y", price_type="open"):
        prices = self.history(start, end)
        historical_returns = []
        for i in range(len(prices) - 1):
            historical_returns.append((prices[i + 1] - prices[i]) / prices[i])
            
        return pd.DataFrame(historical_returns, columns=[self.ticker], index=prices.index[:-1])

    def set_date(self, date):
        self.date = date

    def arima(self, p, d, q, end=date.today(), time_frame="1Y"):
        """
        Use confidence of ARIMA to define how many stocks to buy/sell
        TODO: clean up
        Use information criterion to optimize model, maybe use KL divergence
        """
        
        from statsmodels.tsa.arima_model import ARIMA
        import matplotlib.pyplot as plt
        from pandas.plotting import lag_plot
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        hist = self.history(end=end, time_frame=time_frame)
        # from pandas.plotting import autocorrelation_plot
        # autocorrelation_plot(hist)
        # plt.acorr(hist, maxlags=600)
        # plt.show()
        
        plt.figure(figsize=(10,10))
        lag_plot(hist, lag=5)
        plt.title(f"Random {self.ticker} correlation plot")

        train_data, test_data = train_test_split(hist, shuffle=False)
        plt.figure(figsize=(12,7))
        plt.title('Microsoft Prices')
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.plot(train_data, 'blue', label='Training Data')
        plt.plot(test_data, 'green', label='Testing Data')


        
        print(train_data, test_data)
        
        plt.legend()
        # plt.show()
        
        def smape_kun(y_true, y_pred):
            return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

        history = list(train_data)
        predictions = list()

        cash = 100_000
        shares = 0
        
        y_pred = []
        y = []
        
        for t in range(len(test_data)):
            model = ARIMA(history, order=(p,d,q))
            model_fit = model.fit(disp=0)
            print("params", model_fit.params)
            output = model_fit.predict(start=len(history), end=len(history))  # returns single value (difference)
            print(model_fit.forecast(steps=1)) # return array of prediction, stderr, confidence interval
            print(output)
            yhat = max(output)
            price = history[-1]

            predictions.append(yhat)
            obs = test_data[t]
            history.append(obs)

            if yhat > price:
                # price is going up
                amt = cash // price // 10
                if not shares:
                    amt = cash // price
                  
                shares += amt
                cash -= amt*price
                # print(f"bought {amt} @ {price}")
                y_pred.append(1)

            elif yhat < price:
                # price is going down
                amt = cash // price // 10
                if amt > shares:
                    amt = shares
                shares -= amt
                cash += amt*price
                # print(f"sold {amt} @ {price}")
                y_pred.append(0)
                
            if obs < price:
                y.append(0)
            else:
                y.append(1)
                    
            if not t % 5:
                print(f"done {t} of {len(test_data)}")

        print(f"holding return: {(test_data[-1] - test_data[0])/(test_data[0])}")
        final_val = cash + shares*history[-1]
        print(f"trading returns: {(final_val - 100_000)/100_000}")
        
        error = mean_squared_error(test_data, predictions)
        print('Testing Mean Squared Error: %.3f' % error)
        error2 = smape_kun(test_data, predictions)
        print('Symmetric mean absolute percentage error: %.3f' % error2)

        plt.figure(figsize=(12,7))
        plt.plot(test_data.index, predictions, color='green', linestyle='dashed',label='Predicted Price')
        plt.plot(test_data.index, test_data, color='red', label='Actual Price')
        plt.legend()
        plt.title(f'{self.ticker} Prices Prediction')
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()
