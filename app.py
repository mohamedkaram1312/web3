import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to analyze stock and predict next month's price
def analyze_stock(ticker):
    # Load historical data
    data = yf.download(ticker, start="2021-01-01", end="2024-01-01")
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Simple Moving Average (SMA)

    # Drop rows with NaN values (to simplify)
    data = data.dropna()

    # Prepare data for LSTM model
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])

    # Create sequences for LSTM
    X, y = create_sequences(scaled_data, step=10)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build simple LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # Make prediction for next month
    last_sequence = scaled_data[-10:]
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    predicted_price_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

    return data, data['Close'].iloc[-1].item(), predicted_price  # Convert Series to scalar

# Streamlit app
st.title("Simple Stock Price Prediction")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

if st.button("Predict"):
    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            data, last_price, predicted_price = analyze_stock(ticker)

            # Display results
            st.write("### Historical Data")
            st.line_chart(data['Close'])

            st.write("### Prediction")
            st.write(f"**Last Close Price**: ${last_price:.2f}")
            st.write(f"**Predicted Price for Next Month**: ${predicted_price:.2f}")
