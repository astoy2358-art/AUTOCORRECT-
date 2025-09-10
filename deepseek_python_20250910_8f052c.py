# generate_large_stock_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_stock_data(num_symbols=50, days_per_symbol=1000):
    """Generate large stock dataset with multiple symbols"""
    np.random.seed(42)
    all_data = []
    
    # Create multiple stock symbols
    symbols = [f'STK_{i:03d}' for i in range(1, num_symbols + 1)]
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        
        # Base parameters for each stock
        base_price = random.uniform(10, 500)
        volatility = random.uniform(0.01, 0.05)
        trend = random.uniform(-0.0002, 0.0005)
        
        # Generate dates
        start_date = datetime(2010, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(days_per_symbol)]
        
        # Generate price series with random walk + trend
        prices = [base_price]
        for i in range(1, days_per_symbol):
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(1, new_price))  # Ensure price doesn't go below 1
        
        # Create OHLCV data
        for i, date in enumerate(dates):
            open_price = prices[i]
            close_price = prices[i] * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.02))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.02))
            volume = int(np.random.lognormal(12, 1.2))
            
            all_data.append({
                'Symbol': symbol,
                'Date': date.strftime('%Y-%m-%d'),
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'Sector': random.choice(['Technology', 'Healthcare', 'Financial', 
                                       'Consumer', 'Energy', 'Industrial'])
            })
    
    df = pd.DataFrame(all_data)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Save to CSV
    df.to_csv('stock_data_large.csv', index=False)
    print(f"Dataset created with {len(df):,} records")
    print(f"File saved as 'stock_data_large.csv'")
    
    return df

def add_technical_indicators(df):
    """Add technical indicators to the dataset"""
    print("Adding technical indicators...")
    
    # Group by symbol for calculations
    grouped = df.groupby('Symbol')
    
    # Moving Averages
    df['SMA_20'] = grouped['Close'].transform(lambda x: x.rolling(window=20).mean())
    df['SMA_50'] = grouped['Close'].transform(lambda x: x.rolling(window=50).mean())
    df['EMA_12'] = grouped['Close'].transform(lambda x: x.ewm(span=12).mean())
    df['EMA_26'] = grouped['Close'].transform(lambda x: x.ewm(span=26).mean())
    
    # RSI
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = grouped['Close'].transform(calculate_rsi)
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # Bollinger Bands
    df['BB_Middle'] = grouped['Close'].transform(lambda x: x.rolling(window=20).mean())
    bb_std = grouped['Close'].transform(lambda x: x.rolling(window=20).std())
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Stochastic Oscillator
    def stochastic_oscillator(group):
        low_14 = group['Low'].rolling(window=14).min()
        high_14 = group['High'].rolling(window=14).max()
        return 100 * ((group['Close'] - low_14) / (high_14 - low_14))
    
    df['Stochastic_14'] = grouped.apply(stochastic_oscillator).reset_index(level=0, drop=True)
    
    # Volume indicators
    df['Volume_MA_20'] = grouped['Volume'].transform(lambda x: x.rolling(window=20).mean())
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Price changes
    df['Daily_Return'] = grouped['Close'].pct_change()
    df['Volatility_20'] = grouped['Daily_Return'].transform(lambda x: x.rolling(window=20).std())
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Generate the large dataset
if __name__ == "__main__":
    df = generate_large_stock_data(num_symbols=50, days_per_symbol=1000)
    print("Dataset preview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")