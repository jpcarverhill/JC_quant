import yfinance as yf
import pandas as pd
import numpy as np

def percentage_return_classifier(percentage_return):
    if percentage_return > -0.3 and percentage_return <= 0.3:
        return 'Insignificant Change'
    elif percentage_return > 0.3 and percentage_return <= 3:
        return 'Positive Change'
    elif percentage_return > -3 and percentage_return <= -0.3:
        return 'Negative Change'
    elif percentage_return > 3 and percentage_return <= 7:
        return 'Large Positive Change'
    elif percentage_return > -7 and percentage_return <= -3:
        return 'Large Negative Change'
    elif percentage_return > 7:
        return 'Bull Run'
    elif percentage_return <= -7:
        return 'Bear Sell Off'

def single_stock(TICKER: str, start_date: str, end_date: str):
    # Download the data from yahoo finance
    df = yf.download(TICKER, start=start_date, end=end_date)

    # Clean dataframe
    df = df.reset_index()

    # Calculate daily returns and trends
    df['Daily Return'] = df['Adj Close'].pct_change(1) * 100
    df['Daily Return'] = df['Daily Return'].fillna(0)
    df['Trend'] = df['Daily Return'].apply(percentage_return_classifier)

    return df

def multiple_stock(TICKER_LIST: list, start_date: str, end_date: str):
    # Download the data from yahoo finance
    df = yf.download(TICKER_LIST, start=start_date, end=end_date)

    # Clean dataframe
    df = df['Adj Close']
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Calculate daily returns
    daily_returns_df = df.iloc[:, 1:].pct_change() * 100
    daily_returns_df = daily_returns_df.fillna(0)
    daily_returns_df.insert(0, "Date", df['Date'])

    return df, daily_returns_df
