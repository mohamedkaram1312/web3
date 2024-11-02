import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Function to compute technical indicators
def compute_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    for period in [3, 5, 10, 14, 20]:
        data[f'RSI_{period}'] = compute_rsi(data['Close'], period)
    return data

def compute_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to create sequences for LSTM
def create_sequences(data, step=10):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step][0])  # Predicting the next 'Close' price
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
    
    # Check dimensions before reshaping
    if X.shape[0] == 0:
        st.error("Not enough data points to create sequences.")
        return None

    # Split the data into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape input for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predict the next closing price
    last_sequence = scaled_data[-step:]  # Last 'step' data points
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))
    predicted_price = model.predict(last_sequence)
    
    # Inverse transform the predicted price
    predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((predicted_price.shape[0], 7))), axis=1))

    return predicted_price[0][0]

# Streamlit UI
st.title("Stock Price Predictor with Indicators")

ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-31"))

if st.button("Predict"):
    if ticker and start_date and end_date:
        predicted_price = predict_stock_price(ticker, start_date, end_date)
        if predicted_price is not None:
            st.success(f"Predicted price for {ticker} on {end_date}: ${predicted_price:.2f}")
    else:
        st.error("Please enter a valid stock ticker symbol and ensure the dates are valid.")
