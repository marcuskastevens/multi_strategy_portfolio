# Import Strategy Class
from sma_crossover_strategy_class import sma_crossover_strategy

# Utilities
import pickle


# Relevant Libraries
# import yfinance as yf
# from scipy.optimize import minimize
# from scipy.optimize import Bounds
# from backtest_tools import portfolio_tools as pt

# Define the varying assets for each strategy's investment universe
strategy_assets = [['SPY','QQQ']]

# Define the lookback SMA periods
strategy_lookback_windows = [300]

i = 0

# Create and pickle strategies
for assets in strategy_assets:

    for lookback_window in strategy_lookback_windows:

        strategy = sma_crossover_strategy(tickers=assets, lookback_window=lookback_window)

        path = fr'C:\Users\marcu\Documents\Quant\Programming\Strategies\Beta Strategies\sma_crossover_strategy_{i}.pickle'

        with open(path, 'wb') as handler:
            pickle.dump(strategy, handler, protocol=pickle.HIGHEST_PROTOCOL)
            
        i += 1