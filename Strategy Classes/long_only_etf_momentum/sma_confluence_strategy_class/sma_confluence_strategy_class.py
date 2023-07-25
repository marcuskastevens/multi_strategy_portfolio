# Utils
import pandas as pd
import numpy as np
import datetime as dt

# Relevant Libraries
import yfinance as yf
from backtest_tools import portfolio_tools as pt

class sma_confluence_strategy():

    def __init__(self, tickers: list, lookback_window_slow: int, lookback_window_fast: int):

        self.tickers = tickers
        self.lookback_window_slow = lookback_window_slow
        self.lookback_window_fast = lookback_window_fast
    
        # Get investment unverse's daily returns
        self.close_returns, self.open_returns = self.get_returns(tickers)

        # Backtest Strategy
        self.strategy_returns, self.portfolio_weights = self.run(self.close_returns, self.open_returns, self.lookback_window_slow, self.lookback_window_fast)

    def get_returns(self, tickers: list):
        
        # Returns of opening prices
        open_asset_returns = pd.DataFrame()
        
        # Returns of close prices
        close_asset_returns = pd.DataFrame()

        for ticker in tickers:

            prices = yf.download(ticker, start='1970-01-01', end=dt.date.today())
            
            # Get i'th asset's returns
            close_rets = prices['Adj Close'].pct_change().dropna()
            close_rets = close_rets.rename(ticker)
            open_rets = prices['Open'].pct_change().dropna()
            open_rets = open_rets.rename(ticker)

            close_asset_returns = pd.concat([close_asset_returns, close_rets], axis=1)
            open_asset_returns = pd.concat([open_asset_returns, open_rets], axis=1)

        return(close_asset_returns, open_asset_returns)
    
    
    def get_equally_weighted_returns(self, returns):
        """ Function to equally weight returns across assets based on a pd.Series for a given date. This should be used 
            in conjunction with the df.apply method to do concurrent row operations.

        Args:
            returns (pd.Series): Returns across assets on given day.

        Returns:
            pd.Series: Equally weighted returns series.
        """

        return returns / len(returns.dropna())

    def get_binary_positions(self, returns):
        """ Function to assign binary classifications of long/flat positioning based on a return series that was processed by a strategy.

        Args:
            returns (pd.Series): Long position returns series.

        Returns:
            pd.Series: Binary long/flat position assignments.
        """

        return pd.Series([0 if np.isnan(x) else 1 for i, x in returns.items()], index=returns.index)
    
    def run(self, close_returns: pd.DataFrame, open_returns: pd.DataFrame, lookback_window_slow: int, lookback_window_fast: int):
        
        # Generate strategy
        strategy_returns = (open_returns.shift(-2).where((pt.cumulative_returns(close_returns) > pt.cumulative_returns(close_returns).rolling(lookback_window_fast).mean()) & 
                                            (pt.cumulative_returns(close_returns) > pt.cumulative_returns(close_returns).rolling(lookback_window_slow).mean())))

        # Get non-lagged positions
        strategy_positions = (open_returns.where((pt.cumulative_returns(close_returns) > pt.cumulative_returns(close_returns).rolling(lookback_window_fast).mean()) & 
                                            (pt.cumulative_returns(close_returns) > pt.cumulative_returns(close_returns).rolling(lookback_window_slow).mean())))
        
        # Get binary positions & update multistrategy positions pd.DataFrame
        strategy_positions = strategy_positions.apply(lambda x: self.get_binary_positions(x))

        # Equally weight returns on every day
        strategy_returns = strategy_returns.apply(lambda x: self.get_equally_weighted_returns(x), axis=1)
        strategy_returns = strategy_returns.sum(1)

        return (strategy_returns, strategy_positions)

    def performance_analysis(self, start_date = dt.date(1990, 1, 1)):

        # Print Investment Universe
        print(self.tickers)
        
        # Plot Unscaled Strategy Returns
        pt.cumulative_returns(self.strategy_returns.loc[start_date:]).plot(label='Strategy Returns', legend=True)

        # Plot Vol Scaled Strategy Returns
        scaled_strategy_returns = pt.scale_vol(self.strategy_returns.loc[start_date:])
        pt.cumulative_returns(scaled_strategy_returns).plot(label='Strategy Returns - 10% Vol Scaled', legend=True)

        # Plot 1/N Naive Portfolio as Benchmark Returns
        n = len(self.tickers)
        naive_portfolio = pt.scale_vol((self.open_returns.loc[start_date:]/n).sum(1))
        pt.cumulative_returns(naive_portfolio).plot(label='Naive 1/N Returns - 10% Vol Scaled', legend=True)

        # Plot 50/50 Portfolio of Returns
        combined_portfolio = (.5 * scaled_strategy_returns + .5 * naive_portfolio)
        pt.cumulative_returns(combined_portfolio).plot(label='Combined Opt + Naive Portfolio - 10% Vol Scaled', legend=True)

        # Optimized DASR Portfolio Performance
        strategy_performance_summary = pt.performance_summary(scaled_strategy_returns)
        strategy_performance_summary.index.name = 'Strategy Returns'

        # Naive Portfolio Performance
        naive_performance_summary = pt.performance_summary(naive_portfolio)
        naive_performance_summary.index.name = 'Naive 1/N Returns'

        # 50/50 Combined Portfolio Performance
        combined_portfolio_performance_summary = pt.performance_summary(combined_portfolio)
        combined_portfolio_performance_summary.index.name = 'Combined Portfolio'

        # Update attribute values 
        self.strategy_performance_summary = strategy_performance_summary
        self.naive_performance_summary = naive_performance_summary
        self.combined_strategy_performance_summary = combined_portfolio_performance_summary   
        