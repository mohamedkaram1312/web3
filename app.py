import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data function
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        st.error("No data found for the specified ticker and date range.")
        return None
    data = data[['Close', 'High', 'Low', 'Volume']]
    data['RSI_3'] = RSIIndicator(data['Close'], window=3).rsi()
    data['RSI_5'] = RSIIndicator(data['Close'], window=5).rsi()
    data['RSI_14'] = RSIIndicator(data['Close'], window=14).rsi()
    data.dropna(inplace=True)  # Drop NaN values
    return data

# Build and compile LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function
def main():
    st.title("Stock Price Prediction with LSTM")
    ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):")
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")

    if st.button("Predict"):
        data = load_data(ticker, start_date, end_date)
        if data is not None:
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Prepare training data
            training_data_len = int(np.ceil(len(scaled_data) * .95))
            train_data = scaled_data[0:training_data_len, :]

            # Create the x_train and y_train data sets
            x_train = []
            y_train = []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape data to be 3D for LSTM [samples, time steps, features]
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build and train the model
            model = build_model((x_train.shape[1], 1))
            model.fit(x_train, y_train, batch_size=1, epochs=1)

            # Get the last 60 days of data for prediction
            last_60_days = scaled_data[-60:]
            last_60_days = last_60_days.reshape((1, last_60_days.shape[0], 1))

            # Predicting the price
            predicted_price = model.predict(last_60_days)
            predicted_price = scaler.inverse_transform(predicted_price)

            st.success(f"The predicted price for {ticker} is: {predicted_price[0][0]:.2f}")

if __name__ == "__main__":
    main()
