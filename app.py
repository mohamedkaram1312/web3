import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Function to create sequences for LSTM
def create_sequences(data, step=1):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to predict stock prices
def predict_stock_price(ticker):
    # Fetch historical stock data
    data = yf.download(ticker, start="2021-01-01", end="2024-01-01")
    
    # Use only the 'Close' price
    data = data[['Close']]
    data = data.dropna()

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, step=10)

    # Reshape for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split data into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)

    # Predict the stock prices for the next 30 days
    predictions = []
    last_sequence = scaled_data[-10:]  # Last 10 data points
    for _ in range(30):  # Predict for the next 30 days
        last_sequence = last_sequence.reshape((1, last_sequence.shape[0], 1))
        next_price = model.predict(last_sequence)
        predictions.append(next_price[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predicted_prices

# Streamlit UI
st.title("Stock Price Prediction with LSTM")
ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
if st.button("Predict Price"):
    with st.spinner("Fetching data and predicting..."):
        predicted_prices = predict_stock_price(ticker_input)

    st.write(f"**Predicted Prices for the next 30 days for {ticker_input}:**")
    for i, price in enumerate(predicted_prices):
        st.write(f"Day {i + 1}: ${price[0]:.2f}")
