import pandas as pd

from .base import BaseModel
from statsmodels.tsa.arima_model import ARIMA as ARIMABase
from sklearn.metrics import f1_score

class ARIMA(BaseModel):
    def __init__(self, p, d, q):
        """ Has to be backtestable, runnable, should save data. probably start by writing out something that works just to run. """
        self.historical = pd.DataFrame()
        self.predictions = pd.DataFrame(columns=["prediction"])   # time and the prediction at that time, also would like to give the column a name, see how that matters when appending
        self.p, self.d, self.q = p, d, q

    def step(self, datum, column="open"):
        """
        Datum has to be a row froma pandas dataframe
        """
        self.historical = self.historical.append(datum)
        # the model doesn't have enough data to run properly
        if len(self.historical) < 45:
            return None, None, None

        model = ARIMABase(self.historical[column], order=(self.p, self.d, self.q))
        model_fit = model.fit(disp=0, maxiter=1_000)

        # result = model_fit.predict(start=len(self.historical), end=len(self.historical))
        prediction, stderr, conf_intrvl = model_fit.forecast(steps=1)
        prediction = prediction.item()
        now_df = pd.DataFrame([prediction], columns=["prediction"], index=datum.index)
        # self.predictions = pd.concat([self.predictions, now_df], ignore_index=False)
        self.predictions = self.predictions.append(now_df)
        return prediction, stderr, conf_intrvl
    
    def loss(self):
        return f1_score(self.predictions, self.historical["close"])
    
    def backfill_data(self, df):
        # takes data frame and fills in the historical numbers...
        # maybe runs the predictions to populate that variable
        pass
