import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Function to compute technical indicators
def compute_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    # Stochastic Oscillator
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Drop NaN values
    data.dropna(inplace=True)
    
    return data

# Function to predict price direction based on indicators
def predict_price_direction(ticker):
    data = yf.download(ticker, start="2022-01-01", end="2024-11-01")
    data = compute_indicators(data)

    # Check indicators for predicting direction
    last_row = data.iloc[-1]
    price_increase = False

    # Example logic based on indicators
    if last_row['SMA_50'] > last_row['SMA_200']:  # Golden Cross
        price_increase = True
    elif last_row['RSI'] < 30:  # Oversold
        price_increase = True
    elif last_row['MACD'] > 0:  # MACD is positive
        price_increase = True
    elif last_row['%K'] > last_row['%D']:  # Stochastic indicates bullish
        price_increase = True

    return price_increase

# Streamlit app setup
st.title("Stock Price Direction Prediction Using Indicators")

# User input for the ticker
ticker = st.text_input("Enter Ticker Symbol:", "AAPL")

if st.button("Predict Direction"):
    with st.spinner("Analyzing..."):
        direction = predict_price_direction(ticker)

        if direction:
            st.success(f"The predicted direction for {ticker} is an **increase** in price.")
        else:
            st.error(f"The predicted direction for {ticker} is a **decrease** in price.")
