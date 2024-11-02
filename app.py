import streamlit as st 
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate %K and %D
def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=d_window).mean()
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
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

# Function to create sequences for LSTM
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step, 0])  # Target is Close price
    return np.array(X), np.array(y)

# Function to analyze stock and predict next month's price using multiple indicators
def analyze_stock(ticker, start_date, end_date):
    # Load historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI_3'] = calculate_rsi(data, 3)
    data['RSI_5'] = calculate_rsi(data, 5)
    data['RSI_10'] = calculate_rsi(data, 10)
    data['RSI_14'] = calculate_rsi(data, 14)
    data['RSI_20'] = calculate_rsi(data, 20)
    
    # Calculate Stochastic Oscillator
    data = calculate_stochastic_oscillator(data)

    # Calculate MACD
    data = calculate_macd(data)

    # Calculate OBV
    data = calculate_obv(data)

    # Drop rows with NaN values (caused by rolling windows)
    data = data.dropna()

    # Prepare data for scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_50', 'SMA_200', 'EMA_50', 
                                             'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', 
                                             '%K', '%D', 'MACD', 'MACD_signal', 'OBV']])  

    # Create sequences for LSTM model
    X, y = create_sequences(scaled_data, step=10)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # Make prediction for next month
    last_sequence = scaled_data[-10:]  # Last sequence of all features
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))  
    predicted_price_scaled = model.predict(last_sequence)

    # Inverse transform the predicted price
    predicted_price = scaler.inverse_transform(
        np.array([[predicted_price_scaled[0][0]] + [0] * (scaled_data.shape[1] - 1)])
    )

    return data['Close'].iloc[-1].item(), predicted_price[0][0]

# Streamlit app
st.title("Stock Price Prediction with Multiple Indicators")

# User input for ticker and date range
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
start_date = st.date_input("Select Start Date", datetime(2021, 1, 1))
end_date = st.date_input("Select End Date", datetime(2024, 1, 1))

if st.button("Predict"):
    if ticker and start_date < end_date:
        with st.spinner(f"Fetching data for {ticker} from {start_date} to {end_date}..."):
            last_price, predicted_price = analyze_stock(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            # Display prediction results
            st.write("### Prediction")
            st.write(f"**Last Close Price**: ${last_price:.2f}")
            st.write(f"**Predicted Price for Next Month**: ${predicted_price:.2f}")
    else:
        st.error("Please enter a valid ticker and ensure the end date is after the start date.")
