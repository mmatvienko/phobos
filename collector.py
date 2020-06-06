class Collector():
    """
    
    """
    def __init__(self):
        pass

    def ready(self):
        """
        when the the collector is full, return True
        i.e. sma5, only return ready when queue is full
        this is good for running back tests. for example, might need close from yesterday and open from today. will only be ready when both values are set.
        this gets passed into a strategy probably and the str atwill only start when all collectors are ready
        """
        pass
