import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Function to compute MACD and Signal Line
def compute_macd(data):
    # Calculate the MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to analyze stock based on MACD
def analyze_stock(ticker):
    data = yf.download(ticker, start="2022-10-24", end="2024-11-01")
    data = compute_macd(data)

    if len(data) < 10:
        return ticker, "Not enough data"

    # Get the last available MACD and Signal values
    last_macd = data['MACD'].iloc[-1]
    last_signal = data['Signal'].iloc[-1]

    # Determine the trend based on MACD and Signal
    if last_macd > last_signal:
        predicted_movement = 'Increase'
    elif last_macd < last_signal:
        predicted_movement = 'Decrease'
    else:
        predicted_movement = 'No Change'

    return ticker, last_macd, last_signal, predicted_movement

# Streamlit UI
st.title("Stock MACD Movement Prediction")

ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "SUGR.CA")

if st.button("Analyze"):
    ticker, last_macd, last_signal, predicted_movement = analyze_stock(ticker_input)
    if last_macd == "Not enough data":
        st.error(f"Not enough data for {ticker}.")
    else:
        st.success(f"{ticker}: Last MACD: {last_macd:.2f}, Last Signal: {last_signal:.2f}, Predicted Movement: {predicted_movement}")
