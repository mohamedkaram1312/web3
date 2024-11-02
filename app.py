import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Function to compute technical indicators
def compute_indicators(data):
    # Calculate moving averages
    if 'Close' in data.columns:
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
    # Calculate Bollinger Bands
    if 'Close' in data.columns:
        data['BB_High'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
        data['BB_Low'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    
    # Calculate RSI
    if 'Close' in data.columns:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

    # Volume-Based Indicators
    if 'Volume' in data.columns:
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

    # Ensure that we only create Stochastic indicators if there are enough data points
    if 'Low' in data.columns and 'High' in data.columns and len(data) >= 14:
        data['L14'] = data['Low'].rolling(window=14).min()
        data['H14'] = data['High'].rolling(window=14).max()
        data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
        data['%D'] = data['%K'].rolling(window=3).mean()

    return data

# Main function for Streamlit app
def main():
    st.title("Stock Price Prediction")

    ticker = st.text_input("Enter stock ticker:", value="AAPL")
    start_date = st.date_input("Start date:", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End date:", pd.to_datetime("2022-12-31"))

    if st.button("Predict"):
        # Load data
        data = yf.download(ticker, start=start_date, end=end_date)
        data = compute_indicators(data)
        data.dropna(inplace=True)

        # Display data
        st.write(data)

if __name__ == "__main__":
    main()
