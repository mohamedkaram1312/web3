import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Function to compute technical indicators
def compute_indicators(data):
    # Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # RSI for multiple periods
    for period in [3, 5, 10, 14, 20]:
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Drop rows with NaN values
    return data.dropna()

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to analyze stock and predict next month's price
def analyze_stock(ticker, start_date, end_date):
    # Load historical data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Compute indicators
    data = compute_indicators(data)

    # Prepare features and target
    features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', '%K', '%D']]
    target = data['Close'].shift(-1)  # Predicting the next closing price

    # Scale the features and target
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target[:-1].values.reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(features_scaled, step=10)
    y = target_scaled[:len(y)]  # Ensure y matches X

    # Split the data into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Prepare to predict the next month's price
    last_sequence = features_scaled[-10:]  # Last 10 data points
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

    # Predict future price
    future_price = model.predict(last_sequence)
    predicted_price = scaler_y.inverse_transform(future_price)[0][0]

    return data['Close'].iloc[-1], predicted_price

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
