import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
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

# Function to predict stock price
def predict_stock_price(ticker):
    data = yf.download(ticker, start="2021-01-01", end="2024-01-01")
    
    # Prepare the data
    data = data[['Close']]
    data = data.dropna()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    step = 10
    X, y = create_sequences(scaled_data, step)
    
    # Split the data into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape input for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predict the next closing price
    last_sequence = scaled_data[-step:]  # Last 10 data points
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    predicted_price = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price[0][0]

# Streamlit UI
st.title("Stock Price Predictor")
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):")

if st.button("Predict Price"):
    if ticker:
        predicted_price = predict_stock_price(ticker)
        st.success(f"The predicted next closing price for {ticker} is: ${predicted_price:.2f}")
    else:
        st.error("Please enter a valid stock ticker symbol.")
