import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=window).std() * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=window).std() * 2)
    return data

# Function to calculate OBV
def calculate_obv(data):
    data['Volume'] = data['Volume'].astype(float)
    obv = [0]  # Initial OBV
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    return data

# Function to calculate Support and Resistance levels
def calculate_support_resistance(data):
    data['Support'] = data['Close'].rolling(window=20).min()
    data['Resistance'] = data['Close'].rolling(window=20).max()
    return data

# Function to calculate Chaikin Money Flow (CMF)
def calculate_cmf(data, window=20):
    data['Money Flow Volume'] = (data['Close'] - data['Low'] - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
    data['CMF'] = data['Money Flow Volume'].rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    return data

# Function to analyze stock and predict price movement
def analyze_stock(ticker, start_date, end_date):
    # Load historical data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI_3'] = calculate_rsi(data, 3)
    data['RSI_5'] = calculate_rsi(data, 5)
    data['RSI_10'] = calculate_rsi(data, 10)
    data['RSI_14'] = calculate_rsi(data, 14)
    data['RSI_20'] = calculate_rsi(data, 20)
    
    # Calculate MACD
    data = calculate_macd(data)

    # Calculate Bollinger Bands
    data = calculate_bollinger_bands(data)

    # Calculate OBV
    data = calculate_obv(data)

    # Calculate Support and Resistance
    data = calculate_support_resistance(data)

    # Calculate CMF
    data = calculate_cmf(data)

    # Drop rows with NaN values (caused by rolling windows)
    data = data.dropna()

    # Analyze the indicators to predict future price movement
    last_row = data.iloc[-1]
    indicators = {
        'SMA_50': last_row['SMA_50'],
        'EMA_50': last_row['EMA_50'],
        'RSI_3': last_row['RSI_3'],
        'RSI_5': last_row['RSI_5'],
        'RSI_10': last_row['RSI_10'],
        'RSI_14': last_row['RSI_14'],
        'RSI_20': last_row['RSI_20'],
        'MACD': last_row['MACD'],
        'Signal': last_row['Signal'],
        'OBV': last_row['OBV'],
        'CMF': last_row['CMF'],
        '%K': last_row['%K'] if '%K' in data.columns else None,
        '%D': last_row['%D'] if '%D' in data.columns else None,
        'BB_Middle': last_row['BB_Middle'],
        'BB_Upper': last_row['BB_Upper'],
        'BB_Lower': last_row['BB_Lower'],
        'Support': last_row['Support'],
        'Resistance': last_row['Resistance']
    }

    # Logic to predict increase or decrease
    prediction = ""
    if (indicators['RSI_14'] < 30) and (indicators['Close'] < indicators['Support']):
        prediction = "The stock is likely to increase (oversold condition)."
    elif (indicators['RSI_14'] > 70) and (indicators['Close'] > indicators['Resistance']):
        prediction = "The stock is likely to decrease (overbought condition)."
    else:
        prediction = "The stock movement is uncertain."

    return indicators, prediction

# Streamlit app
st.title("Stock Movement Prediction")

# User input for ticker and date range
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
start_date = st.date_input("Select Start Date", pd.to_datetime("2021-01-01"))
end_date = st.date_input("Select End Date", pd.to_datetime("2024-01-01"))

if st.button("Predict"):
    if ticker and isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp) and start_date < end_date:
        with st.spinner(f"Fetching data for {ticker} from {start_date} to {end_date}..."):
            indicators, prediction = analyze_stock(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            # Display prediction results
            st.write("### Indicators:")
            for key, value in indicators.items():
                st.write(f"**{key}**: {value:.2f}" if isinstance(value, float) else f"**{key}**: {value}")

            st.write("### Prediction:")
            st.write(prediction)
    else:
        st.error("Please enter a valid ticker and ensure the end date is after the start date.")
