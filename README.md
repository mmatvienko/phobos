# Project Phobos

### Jun 13
a long with a trading bot. I envision a tool for visualizing information. graph your value at risk. Or, get a stocks historical data, or plot its autocorrelation. could make a pipeline of a neural network and ARIMA. the neural network has loss function of how correlated data is correlated at a certain lag. this is then passed into ARIMA. ARIMA predicts and that is passed back to neural network to get the predicted data.

### Jun 12 
Got some stuff done from earlier TODOs. Need a base model that all models inherit.

### Jun 11
Got the local data caching working... probably. Have to write some tests and experiment with it. To make sure I don't let any extranious calls out. `

### Jun 9 
Seem to be beating buy and hold in a bull market with the randomized microsoft data. beat return by ~1-2% in 315 trading days. Can predict whether the stock will go up or down about >55% (best has been 65%) of the time. I hope my code is right :D
~Want to test this on real data~, so I definitely have to ~get going on the data caching~ system to avoid the multitude of calls to the API.

### Jun 8
Wrote basic ARIMA. Seems to kind of work.
Want to do gridsearch on it, put it in a strategy and backtest is. Seems like it could actually work based on graphs

### Jun 5
Have finished monte carlo & historical VaR. Seems to be working fine, but I am not sure how to test. The move now is add local data caching to avoid multiple calls to the API (and also when testing, I can avoid getting different random data everytime). 
I'd also like to implement some sort of plot to see how VaR is calculated relative to portfolio holdings.
