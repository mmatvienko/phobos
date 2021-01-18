import scipy.stats as st
import pandas as pd
import numpy as np
import logging

from titan.dumb_broker import DumbBroker
from titan.security import Security
from numpy.linalg import inv


class Portfolio():
    """
    Mainly just a dict of positions
    Keep track of cash inserted to calculate net return
    """
    def __init__(self, cash=0, broker=DumbBroker()):
        self.positions = {}
        self.init_cash = cash
        self.cash = cash
        self.net_value = cash
        self.min_cash = cash
        self.broker = broker
        self.stats = pd.DataFrame([], 
            columns=[
                "drawdown",
                "net_value", 
                "net_cash", 
                "net_equity",
            ],
            index=pd.DatetimeIndex([], name="time"),
        )

    def __getitem__(self, ticker):
        if ticker not in self.positions:
            self.positions[ticker] = 0
        return self.positions[ticker]

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        keys = [key for key in self.positions]
        if self._n >= len(keys): 
            raise StopIteration 
        else: 
            self._n += 1
            return keys[self._n-1]

    def save_stats(self, timestamp=None):
        if timestamp is None:
            raise ValueError("Timestamp for save_stats cannot be none right now.")

        net_cash = self.cash
        net_value = self.evaluate(timestamp=timestamp)
        net_equity = net_value - net_cash

        to_save = {
            "net_equity": net_equity,
            "net_value": net_value,
            "net_cash": net_cash,
        }
        self.stats = self.stats.append(to_save, ignore_index=True)   # instead of appending, do an insert instead

    def sharpe_optimal(self, r_f=0, timestamp=None):
        """ calculate sharpe optimal holdings using formula from variance minimization
        given some risk tolernace parameter
        TODO make it so you can set constraints and only have long positions
        """
        x = self.weight_vector(timestamp=timestamp)    # current holdings
        delta = pd.Timedelta('3Y')
        returns = pd.DataFrame()

        for ticker in self.positions:
            hist = Security(ticker).history(
                timestamp - delta, 
                timestamp, 
                price_type='close')

            returns[ticker] = hist.pct_change(1)
    
        V = returns.cov()
        mu = returns.mean()
        mu_hat = mu - r_f    # risk adjusted returns
        port_var = np.dot(x, np.dot(V, x.T))

        opt_pos = np.dot(inv(V), mu_hat)
        sharpe_optimal = opt_pos/sum(opt_pos)
        
        return sharpe_optimal, port_var

    def order(self, ticker:str, amt:int, price:float=None, timestamp:pd.Timestamp=None):
        """ Can go short, can go long based on amt
        """
        if amt*Security(ticker).get_price(timestamp=timestamp) > self.cash:
            return False

        if not amt:
            # on order for 0
            return False

        amt, price = self.broker.order(ticker, amt=amt, price=price, timestamp=timestamp)
        p_str = f"{'SELL -' if amt < 0 else 'BUY +'} {amt}"
        
        if isinstance(price, pd.Series):
            # should never happen anymore
            raise ValueError(f"The results of {p_str} {ticker} shouldn't be a Series")
        if price:
            logging.info(f"{p_str} {ticker} @{price}")
        else:
            logging.info(f"Failed to execute {p_str} {ticker} @{price}")
            return None, None
        
        total_cost = amt*price

        if ticker in self.positions:
            self.positions[ticker] += amt
        else:
            self.positions[ticker] = amt

        logging.info(f"Total cost of transaction is {total_cost}")
        self.cash -= total_cost

        return amt, price

    def close_pos(self, ticker, timestamp:pd.Timestamp=None):
        """ Close out the position, whatever it may be. Meaning the ticker amy will be 0
        """
        # make sure the position actually exists
        if ticker not in self.positions:
            self.positions[ticker] = 0
            return False

        # amount needed to transact to offset current open position
        amt = -self.positions[ticker]
        if not amt:
            return False

        # want to keep trying to sell the fill amount until a valid order comes through
        ATTEMPTS = 5
        for _ in range(ATTEMPTS):
            # TODO handle partial orders
            close_amt, close_price = self.order(ticker, amt, timestamp=timestamp)
            if close_amt > 0:
                logging.info(f"Closed out {close_amt} {ticker} @{close_price}")
                return True
        
        return False

    @classmethod
    def read(self, key):
        """ loads from disk
        key: the unique identifier for the portfolio, to know which one to load
        """
        pass

    def write(self, key):
        """ write to disk
        key: the unique identifier for the portfolio, to know which one to load
        """
        pass

    def net_return(self):
        return self.net_value - self.init_cash
        
    def evaluate(self, timestamp=None):
        """ Save the worth of portfolio to a dataframe with time index"""
        self.net_value = 0
        
        for ticker in self:
            self.net_value += Security(ticker).get_price(timestamp=timestamp) * self[ticker]

        self.net_value += self.cash
        return self.net_value
            
    def add_cash(self, amount, date=None):
        """ Add cash to portfolio to buy more stuff.
        Also save this amount to init_cash, to calc total return
        """
        self.cash += amount
        self.init_cash += amount
        
    def position_vector(self):
        """ Returns vector for portfolio of whatever information the func returns 
        Need the ndmin to be 2 so that we can do matrix math with the vector
        """
        return np.array([self.positions[ticker] for ticker in self.positions], ndmin=2)

    def price_vector(self, timestamp=None):
        return np.array(
                [
                    Security(ticker).get_price(timestamp=timestamp) 
                    for ticker in self.positions
                ],
                ndmin=2
            )
    
    def weight_vector(self, timestamp=None):
        """ Returns vector for portfolio of whatever information the func returns """

        val_vector = self.position_vector()*self.price_vector(timestamp=timestamp)
        return val_vector/np.sum(val_vector)
        
    def return_history(self, end=pd.to_datetime("today"), time_frame="1Y"):
        """
        Get price history
        create dataframe 
        """
        delta = pd.Timedelta(time_frame)
        start = end - delta

        return_history = pd.DataFrame()
        index = None
        
        # get the price history for all positions in portfolio
        for ticker in self.positions:
            hist = Security(ticker).history(start, end, price_type='open')

            return_history[ticker] = hist.pct_change(1)
            
            if index is None:
                index = hist.index
                
        return_history.set_index(index, inplace=True)
        return return_history

    def incremental_VaR(self):
        raise NotImplementedError()
    
    def VaR(self, start=None, end=pd.Timestamp.today(), time_frame="1Y", z_score=2.33, samples=10_000_000, historical=True):
        """
        Given price history do monte carlo to get VaR
        sigma is corresponds to standard deviation in worst case event, i.e.
        2.33 = 99%, 1.63 = 95%, 3 = 99.7% or 1%, 5%, 0.3% chance of the worst case scenario calculated
        should get returns

        
        base incremental value at risk calculator on this
        add Var for bigger time frames too
        """
        if not start:
            start = end - pd.Timedelta(time_frame)

        df = self.return_history(end)  # TODO: handle if these are of different length
                                       # could just drop NaN

        # calculate historical var
        if historical:
            tmp_returns = df.dot(self.weight_vector().T).sum(axis=1)
            historical_var = np.percentile(tmp_returns, st.norm.cdf(z_score))
            return historical_var

        cov_matrix = df.cov()   # covariance matrix
        lambdas, E = np.linalg.eig(cov_matrix)
        L = np.diag(lambdas)

        # we assume real symmetric matrix allowing to express \Sigma = E \Lambda E^T
        np.testing.assert_almost_equal(cov_matrix, E.dot(L).dot(E.T))  # should always be true

        n = L.shape[0]

        mu, sigma = 0, 1        # TODO: research, but probably have to set this to values representative of the original distribution, assuming coumn wise

        # E.T*L*0.5 is our transformation matrix: Tm
        Tm = E.dot(np.sqrt(L))
        uncorrelated_rand = np.random.normal(mu, sigma, (n, samples))
        correlated_rand = Tm.dot(uncorrelated_rand)

        # weight vector scales percent return, based on price and amount of equ.
        total_pl = self.weight_vector(timestamp=end).dot(correlated_rand).sum(axis=0)  # add all the rows
        smallest = np.percentile(total_pl, st.norm.cdf(z_score))
        var = np.mean(smallest)
        return var
