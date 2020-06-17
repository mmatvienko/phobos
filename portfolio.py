import scipy.stats as st
import pandas as pd
import numpy as np

class Portfolio():
    """
    Mainly just a dict of positions
    Keep track of cash inserted to calculate net return
    """
    def __init__(self, cash=0):
        self.positions = {}
        self.init_cash = cash
        self.cash = cash
        self.net_value = cash

    def __getitem__(self, ticker):
        return self.positions[ticker]
        
    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        keys = self.positions.keys()
        if self._n >= len(keys): 
            raise StopIteration 
        else: 
            self._n += 1
            return keys[self._n-1]

    def close_pos(self, ticker, amt=None):
        # if no specified amount, close the whole position
        pos = self.positions[ticker]
        if not amt:
            amt = pos.amount
        price = pos.get_price()    
        pos.amount -= amt
        self.cash += price * amt

        return amt

    def open_pos(self, ticker, amt):
        pos = self[ticker]
        # TODO should do cash check  and report how much actually bought
        pos.amount = amt
        pos.entry_date = date.today()
        pos.entry_price = -1    # doesnt matter, but needs to be recalced (do in Pos)
        
        return amt              # the amount bought

    def net_return(self):
        return self.net_value - self.init_cash
        
    def evaluate(self):
        """ Save the worth of portfolio to a dataframe with time index"""
        self.net_value = 0
        
        for ticker in positions:
            pos = positions[ticker]
            self.net_value += pos.get_price() * pos.get_amount()

        self.net_value += self.cash

        return self.net_value
            
    def add_cash(self, amount):
        """ Add cash to portfolio to buy more stuff.
        Also save this amount to init_cash, to calc total return
        """
        self.cash += amount
        self.init_cash += amount
        
    def add_position(self, position):
        self.positions[position.ticker] = position

    def position_vector(self):
        """ Returns vector for portfolio of whatever information the func returns """
        return np.array([self.positions[ticker].amount for ticker in self.positions], ndmin=2)

    def price_vector(self):
        return np.array([self.positions[ticker].get_price() for ticker in self.positions], ndmin=2)
    
    def weight_vector(self):
        """ Returns vector for portfolio of whatever information the func returns """

        val_vector = self.position_vector()*self.price_vector()
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
            hist = self.positions[ticker].get_returns(start, end)
            
            return_history[ticker] = hist[ticker].to_list()
            
            if index is None:
                index = hist.index
                
        return_history.set_index(index, inplace=True)
        return return_history

    def incremental_VaR(self):
        raise NotImplementedError()
    
    def VaR(self, start=None, end=pd.to_datetime("today"), time_frame="1Y", z_score=2.33, samples=10_000_000, historical=True):
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

        df = self.return_history(end)  # could be of different length

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
        total_pl = self.weight_vector().dot(correlated_rand).sum(axis=0)  # add all the rows
        smallest = np.percentile(total_pl, st.norm.cdf(z_score))
        var = np.mean(smallest)
        return var
