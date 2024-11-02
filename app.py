import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import streamlit as st

# Function to compute technical indicators
def compute_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    for period in [3, 5, 10, 14, 20]:
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=3).mean()

    return data

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Main function to run the Streamlit app
def main():
    st.title("Stock Price Prediction with LSTM")

    # Input for stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., ALUM.CA):", value="ALUM.CA")

    if ticker:
        # Load data
        data = yf.download(ticker, start="2021-01-01", end="2024-11-01")
        if data.empty:
            st.error("No data found for the ticker. Please check the ticker symbol.")
            return

        data = compute_indicators(data)

        # Drop rows with NaN values
        data.dropna(inplace=True)

        # Prepare features and target
        features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', '%K', '%D']]
        target = data['Close'].shift(-1)

        # Scale the features and target
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        features_scaled = scaler_x.fit_transform(features)
        target_scaled = scaler_y.fit_transform(target.values[:-1].reshape(-1, 1))

        # Create sequences
        X, y = create_sequences(features_scaled, step=10)
        y = target_scaled[:len(y)]

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
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Prepare to predict the next month's price
        last_sequence = features_scaled[-10:]
        last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

        # Predict future price
        future_price = model.predict(last_sequence)
        predicted_price = scaler_y.inverse_transform(future_price)[0][0]

        # Display the predicted price
        st.success(f"Predicted price for the next month: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
