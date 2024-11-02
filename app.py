import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to create sequences for LSTM
def create_sequences(data, step=1):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:(i + step), 0])
        y.append(data[i + step, 0])
    return np.array(X), np.array(y)

# Main Streamlit app
st.title("AAPL Stock Price Prediction using LSTM")

# Define the ticker and the time period
ticker = "AAPL"
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# Download data from Yahoo Finance
if st.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Display the data
    st.subheader("Historical Data")
    st.write(data)

    # Use 'Close' price for prediction
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences
    step = st.slider("Select sequence length:", 1, 100, 10)
    X, y = create_sequences(scaled_data, step)

    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # Make predictions
    predictions = model.predict(X)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)

    # Display results
    st.subheader("Predictions")
    st.write(predictions)

    # Plot results
    st.subheader("Original vs Predicted")
    plt.figure(figsize=(14, 5))
    plt.plot(close_prices, color='blue', label='Original Data')
    plt.plot(np.arange(step, step + len(predictions)), predictions, color='red', label='Predictions')
    plt.legend()
    st.pyplot(plt)
