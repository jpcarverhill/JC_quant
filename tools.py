import random
import pandas as pd
import numpy as np

def price_scaling(df):
    scaled_df = df.copy()
    for column in df.columns[1:]:
        scaled_df[column] = scaled_df[column] / scaled_df.loc[0, column]
    return scaled_df

def generate_portfolio_weights(n):
    weights = [random.random() for _ in range(n)]
    return weights/np.sum(weights)

def asset_allocation(df, weights, initial_investment):
    portfolio_df = df.copy()
    df = price_scaling(df)

    for i, stock in enumerate(df.columns[1:]):
        portfolio_df[stock] = weights[i] * df[stock] * initial_investment

    portfolio_df['Portfolio Value [$]'] = portfolio_df[portfolio_df != 'Date'].sum(axis = 1, numeric_only = True)
    portfolio_df['Portfolio Daily Return [%]'] = portfolio_df['Portfolio Value [$]'].pct_change(1) * 100    
    return portfolio_df.fillna(0).round(2)

def simulation_engine(df, weights, initial_investment, rf):
    # Create portfolio_df using the stock, weights, and inital investment
    portfolio_df = asset_allocation(df, weights, initial_investment)
  
    # Calculate the return on the investment 
    return_on_investment = ((portfolio_df['Portfolio Value [$]'][-1:] - portfolio_df['Portfolio Value [$]'][0]) / portfolio_df['Portfolio Value [$]'][0]) * 100
  
    # Daily change of every stock in the portfolio (Note that we dropped the date, portfolio daily worth and daily % returns) 
    portfolio_daily_return_df = portfolio_df.drop(columns = ['Date', 'Portfolio Value [$]', 'Portfolio Daily Return [%]'])
    portfolio_daily_return_df = portfolio_daily_return_df.pct_change(1) 
  
    # Portfolio Expected Return formula
    expected_portfolio_return = np.sum(weights * portfolio_daily_return_df.mean() ) * 252
  
    # Portfolio volatility (risk) formula
    covariance = portfolio_daily_return_df.cov() * 252 
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

    # Calculate Sharpe ratio
    sharpe_ratio = (expected_portfolio_return - rf)/expected_volatility 
    return expected_portfolio_return, expected_volatility, sharpe_ratio, portfolio_df['Portfolio Value [$]'][-1:].values[0], return_on_investment.values[0]
  