import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Function to compute technical indicators
def compute_indicators(data):
    # Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # RSI for multiple periods
    for period in [3, 5, 10, 14, 20]:
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # MACD and Signal Line
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14'])).fillna(0)  # Ensure fillna
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)

    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data

# Function to analyze stock indicators and predict movement
def analyze_stock(ticker):
    data = yf.download(ticker, start="2022-10-24", end="2024-11-01")
    data = compute_indicators(data)

    if len(data) < 10:
        return ticker, "Not enough data"

    # Prepare features based on indicators
    features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', 
                     'MACD', 'Signal', '%K', '%D', 'BB_Middle', 'BB_Upper', 'BB_Lower']].iloc[-10:]

    # Scale the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Determine price movement
    last_price = features_scaled[-1, 0]
    predicted_movement = 'Increase' if last_price > features_scaled[-2, 0] else 'Decrease'

    return ticker, last_price, predicted_movement

# Streamlit UI
st.title("Stock Price Movement Prediction")

ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "SUGR.CA")

if st.button("Analyze"):
    ticker, last_price, predicted_movement = analyze_stock(ticker_input)
    if last_price == "Not enough data":
        st.error(f"Not enough data for {ticker}.")
    else:
        st.success(f"{ticker}: Last Price: {last_price:.2f}, Predicted Movement: {predicted_movement}")
