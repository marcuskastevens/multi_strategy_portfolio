# Import Strategy Class
from sma_confluence_strategy_class import sma_confluence_strategy

# Utilities
import pickle

# Define the varying assets for each strategy's investment universe
strategy_assets = [['SPY','QQQ']]

# Define the lookback SMA periods
strategy_lookback_windows_slow_fast = [[300, 200]]

i = 0

# Create and pickle strategies
for assets in strategy_assets:

    for lookback_window in strategy_lookback_windows_slow_fast:

        strategy = sma_confluence_strategy(tickers=assets, lookback_window_slow=lookback_window[0], lookback_window_fast=lookback_window[1])

        path = fr'C:\Users\marcu\Documents\Quant\Programming\Strategies\Beta Strategies\sma_confluence_strategy_{i}.pickle'

        with open(path, 'wb') as handler:
            pickle.dump(strategy, handler, protocol=pickle.HIGHEST_PROTOCOL)
            
        i += 1