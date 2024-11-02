import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to create sequences for LSTM
def create_sequences(data, step=10):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to analyze stock
def analyze_stock(ticker, start_date, end_date):
    # Fetch data from yfinance
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        return None, None

    # Use only the 'Close' price for simplicity
    data = data[['Close']].dropna()

    # Scaling the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(data_scaled)

    # Train-test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Predict future price for the next 30 days
    future_prices = []
    last_sequence = data_scaled[-10:]  # Last 10 days for prediction
    for _ in range(30):
        last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
        future_price = model.predict(last_sequence)
        future_prices.append(future_price[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[future_price]], axis=1)  # Append the new prediction

    # Inverse scaling to get actual prices
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()
    return future_prices

# Streamlit UI
st.title("Stock Price Prediction")

ticker_input = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.button("Predict Next 30 Days Price"):
    expected_prices = analyze_stock(ticker_input, start_date, end_date)

    if expected_prices is not None:
        st.write(f"**Expected Prices for the next 30 days:**")
        for day, price in enumerate(expected_prices, 1):
            st.write(f"Day {day}: ${price:.2f}")
    else:
        st.write(f"No data available for {ticker_input} from {start_date} to {end_date}.")
