import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import talib

# Function to compute technical indicators
def compute_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI_3'] = talib.RSI(data['Close'], timeperiod=3)
    data['RSI_5'] = talib.RSI(data['Close'], timeperiod=5)
    data['RSI_10'] = talib.RSI(data['Close'], timeperiod=10)
    data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)
    data['RSI_20'] = talib.RSI(data['Close'], timeperiod=20)
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

# Function to predict stock prices
def predict_stock_price(ticker, start_date, end_date):
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    data = compute_indicators(data)
    
    # Prepare the dataset
    feature_columns = ['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20']
    scaled_data = data[feature_columns].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(scaled_data)
    
    # Create sequences
    time_step = 60  # Use 60 days to predict
    X, y = create_dataset(scaled_data, time_step)
    
    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(32))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32)

    # Predict the next 30 days
    last_60_days = scaled_data[-time_step:].reshape(1, time_step, len(feature_columns))
    predictions = []
    
    for _ in range(30):  # Predict for the next 30 days
        predicted_price = model.predict(last_60_days)[0][0]
        predictions.append(predicted_price)
        
        # Update last_60_days with the new prediction
        last_60_days = np.append(last_60_days[:, 1:, :], [[predicted_price] + last_60_days[0, -1, 1:].tolist()], axis=1)

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions

# Streamlit UI
st.title("Stock Price Predictor with Indicators")
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-31"))

if st.button("Predict"):
    if ticker:
        predictions = predict_stock_price(ticker, start_date, end_date)
        st.write(f"Predicted prices for the next 30 days for {ticker}:")
        st.line_chart(predictions)
    else:
        st.error("Please enter a valid stock ticker symbol.")
