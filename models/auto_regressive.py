import pandas as pd

from .base import BaseModel
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import f1_score

class ARIMA(BaseModel):
    def __init__(self, p, d, q):
        """ Has to be backtestable, runnable, should save data. probably start by writing out something that works just to run. """
        self.historical = pd.DataFrame()
        self.predictions = pd.DataFrame()   # time and the prediction at that time 
        self.p, self.d, self.q = p, d, q

    def step(self, datum, column="open"):
        """
        Datum has to be a row froma pandas dataframe
        """
        self.historical.append(datum)
        model = ARIMA(self.historical[column], order(self.p, self.d, self.q))
        model_fit = model.fit(disp=0, maxiter=1_000)

        # result = model_fit.predict(start=len(self.historical), end=len(self.historical))
        prediction, stderr, conf_intrvl = model_fit.forecast(steps=1)
        self.predictions.append(max(prediction))

        return prediction, stderr, conf_intrvl
    
    def loss(self):
        return f1_score(self.predictions, self.historical)

    def backfill_data(self, df):
        # takes data frame and fills in the historical numbers...
        # maybe runs the predictions to populate that variable
        pass
