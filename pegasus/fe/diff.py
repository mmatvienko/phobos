import numpy as np

# this is a test for feature engineering
# will be first order differencing
class Diff():
    def __init__(self, degree=1):
        self.degree = degree
        self.prev_vals = []

    def step(self, val):
        assert len(self.prev_vals) <= self.degree
        if len(self.prev_vals) < self.degree:
            self.prev_vals.append(val)
            return np.nan
        
        self.prev_vals.append(val)
        res = self.prev_vals[1] - self.prev_vals[0]
        del self.prev_vals[0]

        return res

        