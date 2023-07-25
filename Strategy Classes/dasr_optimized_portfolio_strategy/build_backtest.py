# Import Strategy Class
from dasr_optimized_portfolio_strategy_class import dasr_optimized_portfolio

# Utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pickle


# Relevant Libraries
# import yfinance as yf
# from scipy.optimize import minimize
# from scipy.optimize import Bounds
# from backtest_tools import portfolio_tools as pt

# Define the varying assets for each strategy's investment universe
strategy_assets = [ ['SPY','QQQ', 'DBC', 'TLT', 'GLD'],
                    ['SPY','QQQ', 'GLD', 'TLT', 'DBC', 'XLE', 'EEM'],
                    ['SPY', 'GLD', 'TLT', 'DBC', 'JNK', 'EEM'],
                    ['SPY', 'QQQ', 'TLT', 'DBC']
                  ]

# Define the varying rebalancing frequencies for each strategy
strategy_rebal_freq = [20, 10]

i = 0

# Create and pickle strategies
for assets in strategy_assets:

    for rebal_freq in strategy_rebal_freq:

        strategy = dasr_optimized_portfolio(tickers=assets, lookback_window=rebal_freq, rebal_freq=rebal_freq)

        path = fr'C:\Users\marcu\Documents\Quant\Programming\Strategies\Beta Strategies\dasr_optimized_portfolio_strategy_long_only_{i}.pickle'

        with open(path, 'wb') as handler:
            pickle.dump(strategy, handler, protocol=pickle.HIGHEST_PROTOCOL)
            
        i += 1