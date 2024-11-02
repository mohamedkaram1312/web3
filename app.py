import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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

    # MACD and Signal Line
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Volume-Based Indicators
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['CMF'] = (((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / 
                   (data['High'] - data['Low']) * data['Volume']).rolling(window=20).mean()

    # Stochastic Oscillator
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)

    # Support and Resistance Levels
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Resistance'] = data['High'].rolling(window=20).max()

    return data

# Function to create sequences for LSTM
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Streamlit app
st.title("Stock Price Prediction App")

# User input for ticker
ticker = st.text_input("Enter Stock Ticker (e.g., 'AAPL', 'GOOGL', 'ALUM.CA'):", "ALUM.CA")

# Load data and make predictions
if st.button("Predict"):
    data = yf.download(ticker, start="2022-10-24", end="2024-10-20")
    if data.empty:
        st.error("No data found for the provided ticker symbol. Please try another symbol.")
    else:
        data = compute_indicators(data)

        # Drop rows with NaN values
        data.dropna(inplace=True)

        # Prepare features and target
        features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', 
                         'MACD', 'Signal', 'OBV', 'CMF', '%K', '%D', 'BB_Middle', 'BB_Upper', 
                         'BB_Lower', 'Support', 'Resistance']]
        target = data['Close'].shift(-1)  # Predicting the next closing price

        # Scale the features and target
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        features_scaled = scaler_x.fit_transform(features)
        target_scaled = scaler_y.fit_transform(target.values[:-1].reshape(-1, 1))

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
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Prepare to predict the next month's price
        last_sequence = features_scaled[-10:]  # Last 10 data points
        last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

        # Predict future price
        future_price = model.predict(last_sequence)
        predicted_price = scaler_y.inverse_transform(future_price)[0][0]

        # Plot historical and predicted prices
        st.subheader(f'Stock Price Prediction for {ticker}')
        plt.figure(figsize=(14, 7))
        plt.plot(data['Close'], label='Historical Prices', color='blue')
        plt.scatter(data.index[-1] + pd.Timedelta(days=30), predicted_price, color='red', s=100, 
                    label='Predicted Price Next Month', zorder=5)
        plt.title(f'Stock Price Prediction for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        # Show plot in Streamlit
        st.pyplot(plt)

        # Print the predicted price
        st.write(f"Predicted price for the next month: {predicted_price:.2f}")
