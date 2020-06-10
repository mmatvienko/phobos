# Project Phobos

### Jun 5
Have finished monte carlo & historical VaR. Seems to be working fine, but I am not sure how to test. The move now is add local data caching to avoid multiple calls to the API (and also when testing, I can avoid getting different random data everytime). 
I'd also like to implement some sort of plot to see how VaR is calculated relative to portfolio holdings.

### Jun 8
Wrote basic ARIMA. Seems to kind of work.
Want to do gridsearch on it, put it in a strategy and backtest is. Seems like it could actually work based on graphs

### Jun 9 
Seem to be beating buy and hold in a bull market with the randomized microsoft data. beat return by ~1-2% in 315 trading days. Can predict whether the stock will go up or down about >55% (best has been 65%) of the time. I hope my code is right :D
Want to test this on real data, so I definitely have to get going on the data caching system to avoid the multitude of calls to the API.
