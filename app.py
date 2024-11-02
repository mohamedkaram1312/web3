import streamlit as st
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Function to create sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step, 0])  # Use only the Close price as target
    return np.array(X), np.array(y)

# Function to analyze stock and predict next month's price using indicators
def analyze_stock(ticker, start_date, end_date):
    # Load historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day SMA
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day SMA
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()  # 50-day EMA

    # Drop rows with NaN values (due to rolling window)
    data = data.dropna()

    # Scale data (Close, SMA_50, SMA_200, EMA_50)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_50', 'SMA_200', 'EMA_50']])

    # Create sequences for LSTM
    X, y = create_sequences(scaled_data, step=10)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # Make prediction for next month
    last_sequence = scaled_data[-10:]  # Last sequence of all features
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 4))  # 4 features in input
    predicted_price_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(np.array([[predicted_price_scaled[0][0], 0, 0, 0]]))[0][0]

    return data['Close'].iloc[-1].item(), predicted_price

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
