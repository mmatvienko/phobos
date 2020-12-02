class BaseModel():
    def __init__(self):
        pass

    def step(self, time_datum):
        """ Takes the next piece of time data and returns a prediction.
        Perhaps X amount of predictions based on user preference
        """
        raise NotImplementedError()

    def loss(self):
        """ Could be the most basic thing like accuracy or F1.
        """
        raise NotImplementedError()
