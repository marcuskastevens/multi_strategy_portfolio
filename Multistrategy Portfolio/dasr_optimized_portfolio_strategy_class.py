# Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Relevant Libraries
import yfinance as yf
from scipy.optimize import minimize
from scipy.optimize import Bounds
from backtest_tools import portfolio_tools as pt

class dasr_optimized_portfolio():

    def __init__(self, tickers: list, lookback_window: int, rebal_freq: int):

        self.tickers = tickers
        self.lookback_window = lookback_window
        self.rebal_freq = rebal_freq
    
        # Get investment unverse's daily returns
        self.close_returns, self.open_returns = self.get_returns(tickers)

        # Backtest Strategy
        self.strategy_returns, self.portfolio_weights = self.run(self.close_returns, self.open_returns, self.lookback_window, self.rebal_freq)

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

    
    def portfolio_dasr(self, w: pd.Series, returns: pd.DataFrame) ->  float:
        """ Computes DASR of weighted portfolio.

        Args:
            betas (pd.Series): daily expected returns from normalized linear regression.
            squared_residuals (pd.DataFrame): squared error from OLS regression.
            w (pd.Series): portfolio weights.

        Returns:
            float: _description_
        """

        # Get weighted returns
        returns = (returns * w).sum(1)
        
        # Get weighted portfolio DASR
        portfolio_dasr = pt.drift_adjusted_sharpe_ratio(returns.dropna())
        
        # Return negative DASR for portfolio optimization
        return (-portfolio_dasr)

    def max_dasr_optimization(self, returns: pd.DataFrame) -> pd.Series:
        """ Executes constrained convex portfolio optimization to generate optimal
            DASR asset weights.

        Args:
            returns (pd.DataFrame): Asset returns.

        Returns:
            pd.Series: Optimal DASR portfolio weights.
        """

        n = len(returns.columns)

        # Initial guess is naive 1/n portfolio
        w = np.array([1 / n] * n)

        # Max position size L/S
        bounds = Bounds(-.5, .5)

        constraints =  [# Weights Constraint
                        {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1},
                        # {"type": "eq", "fun": lambda w: np.sum(w) - 0},
                        ]

        # Get optimized weights
        w = pd.Series(minimize(self.portfolio_dasr, 
                                w,
                                args=(returns), 
                                method='SLSQP',
                                bounds = bounds,
                                constraints=constraints)['x'],
                    index=returns.columns
                    )
        
        return w
    
    def run(self, close_returns: pd.DataFrame, open_returns: pd.DataFrame, lookback_window: int, rebal_freq: int):
        
        # Declare weights hash table
        w = {}

        # Get optimized weights at rebal_freq intervals
        for date in close_returns.index[::rebal_freq]:
            w[date] = self.max_dasr_optimization(returns=close_returns.loc[:date].tail(lookback_window))

        # Convert Hash Table to DataFrame
        indices_df = pd.DataFrame(index=close_returns.index)
        w = pd.concat([indices_df, pd.DataFrame(w).T], axis=1).ffill().dropna()

        # Get strategy returns
        strategy_returns = (open_returns*w.shift(2)).sum(1).dropna()

        return (strategy_returns, w)

    def performance_analysis(self, start_date = dt.date(2004, 1, 1)):

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
        