import numpy as np
import pandas as pd
import yfinance as yf
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

    return data

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

# Function to analyze a single stock
def analyze_stock(ticker):
    data = yf.download(ticker, start="2021-10-24", end="2024-10-31")
    data = compute_indicators(data)

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Check if there's enough data
    if len(data) < 10:  # Check if there are at least 10 rows left after dropping NaNs
        print(f"Not enough data for {ticker}.")
        return ticker, None, None, None

    # Prepare features and target
    features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20']]
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
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Prepare to predict the next month's price
    last_sequence = features_scaled[-10:]  # Last 10 data points
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

    # Predict future price
    future_price = model.predict(last_sequence)
    predicted_price = scaler_y.inverse_transform(future_price)[0][0]
    last_actual_price = data['Close'].iloc[-1]

    # Calculate expected increase
    percentage_increase = ((predicted_price - last_actual_price) / last_actual_price) * 100

    return ticker, last_actual_price, predicted_price, percentage_increase

# List of tickers to analyze
tickers = ['ALUM.CA']  # Add your desired tickers here

# Initialize the results list
results = []

# Analyze each ticker and gather results
for ticker in tickers:
    ticker, last_price, predicted_price, percentage_increase = analyze_stock(ticker)
    if percentage_increase is not None:  # Only append if there is a valid percentage increase
        results.append((ticker, last_price, predicted_price, percentage_increase))

# Sort results by expected increase, filtering out None values
sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

# Print sorted results
for ticker, last_price, predicted_price, percentage_increase in sorted_results:
    print(f"{ticker}: Last Price: {last_price:.2f}, Predicted Price: {predicted_price:.2f}, Expected Increase: {percentage_increase:.2f}%")
