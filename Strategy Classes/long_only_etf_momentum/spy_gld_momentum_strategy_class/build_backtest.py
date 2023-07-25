# Import Strategy Class
from spy_gld_momentum_strategy_class import spy_gld_momentum_strategy

# Utilities
import pickle

# Build strategy
strategy = spy_gld_momentum_strategy()

# Pickle strategy
path = fr'C:\Users\marcu\Documents\Quant\Programming\Strategies\Beta Strategies\spy_gld_momentum_strategy_0.pickle'

with open(path, 'wb') as handler:
    pickle.dump(strategy, handler, protocol=pickle.HIGHEST_PROTOCOL)
    
