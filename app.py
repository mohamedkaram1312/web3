import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Function to compute MACD and OBV
def compute_indicators(data):
    # Calculate MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate OBV
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

    return data

# Function to analyze stock based on indicators
def analyze_stock(ticker):
    data = yf.download(ticker, start="2022-10-24", end="2024-11-01")
    data = compute_indicators(data)

    if len(data) < 10:
        return ticker, "Not enough data"

    # Get the last available MACD, Signal, and OBV values
    last_macd = data['MACD'].iloc[-1]
    last_signal = data['Signal'].iloc[-1]
    last_obv = data['OBV'].iloc[-1]

    # Determine the trend based on MACD and Signal
    if last_macd > last_signal:
        predicted_movement = 'Increase'
    elif last_macd < last_signal:
        predicted_movement = 'Decrease'
    else:
        predicted_movement = 'No Change'

    return ticker, last_macd, last_signal, last_obv, predicted_movement

# Streamlit UI
st.title("Stock Indicator Movement Prediction")

ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "SUGR.CA")

if st.button("Analyze"):
    ticker, last_macd, last_signal, last_obv, predicted_movement = analyze_stock(ticker_input)
    if last_macd == "Not enough data":
        st.error(f"Not enough data for {ticker}.")
    else:
        st.success(f"{ticker}: Last MACD: {last_macd:.2f}, Last Signal: {last_signal:.2f}, "
                    f"Last OBV: {last_obv:.2f}, Predicted Movement: {predicted_movement}")
