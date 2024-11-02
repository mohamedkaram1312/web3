import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to create dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to predict stock prices
def predict_stock_price(ticker, start_date, end_date):
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create dataset
    X, y = create_dataset(scaled_data, time_step=60)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32)

    # Predict the next 30 days
    last_60_days = scaled_data[-60:].reshape(1, -1, 1)
    predictions = []

    for _ in range(30):
        predicted_price = model.predict(last_60_days)[0][0]
        predictions.append(predicted_price)
        new_data = np.array([[predicted_price]])
        last_60_days = np.append(last_60_days[:, 1:, :], new_data.reshape(1, 1, 1), axis=1)

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Calculate the average of the predicted prices
    average_price = np.mean(predicted_prices)

    return predicted_prices.flatten(), average_price

# Streamlit UI
st.title("Stock Price Predictor")
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-31"))

if st.button("Predict"):
    if ticker:
        try:
            predictions, average_price = predict_stock_price(ticker, start_date, end_date)
            st.write(f"Predicted Prices for the Next 30 Days for {ticker}:")
            st.line_chart(predictions)
            st.write(f"The expected average price over the next 30 days is: {average_price:.2f} KZA")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid stock ticker symbol.")
