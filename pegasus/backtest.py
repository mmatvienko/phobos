import pandas as pd

""" Backtesting is like a runner, but just on historical data """

def backtest_strategy(strategy, start_date, end_date):
    """
    TODO: move to its own file along with backtest_var
    takes a strat and backtests it
    not sure if it should be for a ticker or for portfolio
    """
    for date in pd.bdate_range(start_date, end_date):
        print("\n", date)
        strategy.step_portfolio(date=date)

    # here would be a good place for metrics and graphs
    # how much money could have been made from just holding or something
    # write the results to a file or something
    
    final_value = strategy.portfolio.evaluate(end_date)
    print(f"the minimum cash held: {strategy.portfolio.min_cash}")
    print(f"started with $100,000 on {start_date}, and finished with {final_value} on {end_date}")

def backtest_var(portfolio, start_date, end_date):
    """
    Takes portfolio. then just without strategy, sees what the
    VaR is for every day, plots it, and then also plots actual returns on top
    """
    pass
