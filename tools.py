import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

def VaR_historical(df, confidence_level = 0.95):
    # Historical Method
    returns = df['Adj Close'].pct_change().dropna()
    VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)

    # Plot the historical returns and VaR threshold
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(VaR_historical, color='red', linestyle='--', label=f'VaR (95%): {VaR_historical:.2%}')
    plt.title('Historical Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def VaR_variance_covariance(df, confidence_level = 0.95):
    # Variance-Covariance
    returns = df['Adj Close'].pct_change().dropna()

    # Calculate the mean and standard deviation of returns
    mean_return = np.mean(returns)
    std_dev = np.std(returns)

    # Calculate the VaR at XX% confidence level using the Z-score
    z_score = norm.ppf(1 - confidence_level)
    VaR_variance_covariance = mean_return + z_score * std_dev

    print(f"Variance-Covariance VaR (95% confidence level): {VaR_variance_covariance:.2%}")

    # Plot the normal distribution and VaR threshold
    plt.figure(figsize=(10, 6))
    x = np.linspace(mean_return - 3*std_dev, mean_return + 3*std_dev, 1000)
    y = norm.pdf(x, mean_return, std_dev)
    plt.plot(x, y, label='Normal Distribution')
    plt.axvline(VaR_variance_covariance, color='red', linestyle='--', label=f'VaR (95%): {VaR_variance_covariance:.2%}')
    plt.fill_between(x, 0, y, where=(x <= VaR_variance_covariance), color='red', alpha=0.5)
    plt.title('Normal Distribution of Returns with VaR Threshold')
    plt.xlabel('Returns')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def VaR_monte_carlo(df, confidence_level, num_simulations, simulation_horizon, initial_investment):
    returns = df['Adj Close'].pct_change().dropna()

    # Simulate future returns using Monte Carlo
    simulated_returns = np.random.normal(np.mean(returns), np.std(returns), (simulation_horizon, num_simulations))

    # Calculate the simulated portfolio values
    portfolio_values = initial_investment * np.exp(np.cumsum(simulated_returns, axis=0))

    # Calculate the portfolio returns
    portfolio_returns = portfolio_values[-1] / portfolio_values[0] - 1

    # Calculate the VaR at 95% confidence level
    VaR_monte_carlo = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    print(f"Monte Carlo VaR (95% confidence level): {VaR_monte_carlo:.2%}")

    # Plot the distribution of simulated portfolio returns and VaR threshold
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(VaR_monte_carlo, color='red', linestyle='--', label=f'VaR (95%): {VaR_monte_carlo:.2%}')
    plt.title('Simulated Portfolio Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Value at Risk
# https://medium.com/@serdarilarslan/value-at-risk-var-and-its-implementation-in-python-5c9150f73b0e