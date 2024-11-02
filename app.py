import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Function to compute technical indicators
def compute_indicators(data):
    # Calculate Simple Moving Average (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate Exponential Moving Average (EMA)
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # Calculate RSI for multiple periods
    for period in [3, 5, 10, 14, 20]:
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    return data

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to predict stock price
def predict_stock_price(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Compute indicators
    data = compute_indicators(data)
    data = data.dropna()

    # Prepare features and target
    features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20']]
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
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
st.title("Stock Price Predictor with Indicators")

ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.button("Predict Price"):
    if ticker and start_date < end_date:
        predicted_price = predict_stock_price(ticker, start_date, end_date)
        st.success(f"The predicted next closing price for {ticker} is: ${predicted_price:.2f}")
    else:
        st.error("Please enter a valid stock ticker symbol and ensure that the start date is before the end date.")
