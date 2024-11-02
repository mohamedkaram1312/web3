import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import streamlit as st

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to analyze a single stock
def analyze_stock(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Check if there's enough data
    if len(data) < 10:  # Check if there are at least 10 rows left after dropping NaNs
        st.error(f"Not enough data for {ticker}.")
        return ticker, None, None, None

    # Prepare features and target
    features = data[['Close']]  # Using only the 'Close' price
    target = data['Close'].shift(-1)  # Predicting the next closing price

    # Scale the features and target
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.values[:-1].reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(features_scaled, step=60)
    y = target_scaled[:len(y)]  # Ensure y matches X

    # Split the data into train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(350, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(350, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

    # Prepare to predict the next month's price
    last_sequence = features_scaled[-60:]  # Last 60 data points
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

    # Predict future price
    future_price = model.predict(last_sequence)
    predicted_price = scaler_y.inverse_transform(future_price)[0][0]
    last_actual_price = data['Close'].iloc[-1]

    # Calculate expected increase
    percentage_increase = ((predicted_price - last_actual_price) / last_actual_price) * 100

    return ticker, last_actual_price, predicted_price, percentage_increase

# Streamlit app
st.title('Stock Price Prediction with LSTM')
ticker_input = st.text_input("Enter Stock Ticker (e.g., CCAP.CA):", "CCAP.CA")
start_date = st.date_input("Select Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("Select End Date", pd.to_datetime("2023-10-23"))

if st.button('Predict'):
    ticker, last_price, predicted_price, percentage_increase = analyze_stock(ticker_input, start_date, end_date)
    if predicted_price is not None:
        st.success(f"{ticker}: Last Price: {last_price:.2f}, Predicted Price: {predicted_price:.2f}, Expected Increase: {percentage_increase:.2f}%")
    else:
        st.error("Error in prediction.")
