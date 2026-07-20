import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_real_market_data(start_date: str = "2023-01-01", end_date: str = None, cache_dir: str = "data/cache"):
    """
    Downloads historical data for SPY, ^VIX, and top 5 SPY constituents.
    Calculates 30-day realized volatility as a proxy for constituent IV.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"yfinance_data_{start_date}_{end_date}.parquet")
    
    if os.path.exists(cache_file):
        print(f"[INFO] Loading cached real market data from {cache_file}")
        return pd.read_parquet(cache_file)
        
    print(f"[INFO] Fetching real market data from Yahoo Finance ({start_date} to {end_date})...")
    
    # SPY as our index, VIX for index implied vol, and top 5 constituents
    tickers = ["SPY", "^VIX", "AAPL", "MSFT", "NVDA", "AMZN", "META"]
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("Failed to fetch data from yfinance.")
    
    # Extract adjusted close prices
    # yfinance returns a MultiIndex column DataFrame when multiple tickers are passed
    adj_close = data['Close'] # Using Close since Adj Close is sometimes missing or same in newer yfinance versions
    
    # Create the final dataframe we'll use for the backtest
    df = pd.DataFrame(index=adj_close.index)
    
    # 1. Index Price and Index IV (VIX)
    df['index_price'] = adj_close['SPY']
    # VIX is quoted in whole numbers (e.g., 20.5 for 20.5%). Convert to decimal.
    df['index_iv'] = adj_close['^VIX'] / 100.0
    
    # 2. Constituent Prices and Approximated IVs
    constituents = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    
    # Calculate 30-day rolling realized volatility (annualized)
    # RV = std(daily_returns) * sqrt(252)
    returns = adj_close[constituents].pct_change()
    realized_vol = returns.rolling(window=30).std() * (252 ** 0.5)
    
    for i, ticker in enumerate(constituents):
        df[f'c{i}_price'] = adj_close[ticker]
        # Approximate IV = 30-day Realized Vol + 2% Variance Premium
        # Forward fill the first 30 days which will be NaN due to rolling window,
        # or just backfill.
        df[f'c{i}_iv'] = realized_vol[ticker] + 0.02
        
    # Drop the first 30 rows where RV is NaN
    df = df.dropna()
    
    # Reset index to have 'Date' as a column instead of index, to match our synthetic data format
    df = df.reset_index()
    df.rename(columns={'Date': 'date'}, inplace=True)
    df['t'] = df.index # Add a time step column
    
    print(f"[INFO] Saving fetched data to {cache_file}")
    df.to_parquet(cache_file)
    
    return df

if __name__ == "__main__":
    df = fetch_real_market_data(start_date="2023-01-01", end_date="2024-01-01")
    print(df.head())
    print(f"Data shape: {df.shape}")
