import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

def compute_indicators(data):
    # Ensure that the data has necessary columns
    if 'Close' not in data.columns or 'High' not in data.columns or 'Low' not in data.columns:
        raise ValueError("Data does not contain required columns.")
    
    # Calculate indicators with checks
    try:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # Compute RSI for multiple periods
        for period in [3, 5, 10, 14, 20]:
            delta = data['Close'].diff(1)
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)  # Prevent division by zero
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Compute MACD
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Compute OBV
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        
        # Compute CMF
        data['CMF'] = (((data['Close'] - data['Low']) - (data['High'] - data['Close'])) /
                       (data['High'] - data['Low']).replace(0, np.nan) * data['Volume']).rolling(window=20).mean()

        # Calculate L14 and H14
        data['L14'] = data['Low'].rolling(window=14, min_periods=1).min()
        data['H14'] = data['High'].rolling(window=14, min_periods=1).max()

        # Calculate %K and handle potential division by zero
        data['%K'] = np.where(
            (data['H14'] - data['L14']) == 0,
            np.nan,  # Assign NaN where division is not possible
            100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
        )
        data['%D'] = data['%K'].rolling(window=3).mean()

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
        data['Support'] = data['Low'].rolling(window=20).min()
        data['Resistance'] = data['High'].rolling(window=20).max()

    except Exception as e:
        st.error(f"Error computing indicators: {e}")
    
    return data

def load_data(ticker, start_date, end_date):
    # Fetch the stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.warning("No data available for the selected ticker and date range.")
        return None
    data['Volume'] = data['Volume'].astype(float)  # Ensure volume is float
    return data

def main():
    st.title("Stock Analysis App")
    
    # User input for stock ticker and date range
    ticker = st.text_input("Enter stock ticker (e.g., AAPL):")
    start_date = st.date_input("Start date:")
    end_date = st.date_input("End date:")
    
    if st.button("Analyze"):
        if ticker:
            data = load_data(ticker, start_date, end_date)
            if data is not None:
                data = compute_indicators(data)
                data.dropna(inplace=True)  # Remove any remaining NaN rows
                st.write(data.tail())  # Display the last few rows of data
        else:
            st.warning("Please enter a valid ticker symbol.")

if __name__ == "__main__":
    main()
