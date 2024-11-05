import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from ta.momentum import RSIIndicator, TSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel

# Function to load and preprocess data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close', 'High', 'Low', 'Volume']]
    # Calculate indicators
    data['RSI_3'] = RSIIndicator(data['Close'], window=3).rsi()
    data['RSI_5'] = RSIIndicator(data['Close'], window=5).rsi()
    data['RSI_14'] = RSIIndicator(data['Close'], window=14).rsi()
    data['TSI'] = TSIIndicator(data['Close']).tsi()
    data['WilliamsR'] = WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data['Stochastic'] = StochasticOscillator(close=data['Close'], high=data['High'], low=data['Low']).stoch()
    data['SMA_10'] = SMAIndicator(data['Close'], window=10).sma_indicator()
    data['EMA_10'] = EMAIndicator(data['Close'], window=10).ema_indicator()
    data['MACD'] = MACD(data['Close']).macd_diff()
    bollinger = BollingerBands(close=data['Close'])
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()
    data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    vwap = VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['VWAP'] = vwap.volume_weighted_average_price()
    data['Volatility'] = data['Close'].rolling(window=20).std()
    data = data.dropna()
    return data

# Function to build and train the model
def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    return model

# Main Streamlit app
def main():
    st.title("Stock Price Prediction App")
    
    ticker = st.text_input("Enter stock ticker (e.g., 'ABUK.CA'):", value='ABUK.CA')
    start_date = st.date_input("Start date:", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End date:", pd.to_datetime("2024-11-05"))

    if st.button("Predict"):
        with st.spinner("Loading data..."):
            data = load_data(ticker, start_date, end_date)
            st.write(data.tail())  # Display the last few rows of the data
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Prepare training dataset
            sequence_length = 60
            future_target = 14  # Predict the price 14 days ahead
            X_train, y_train = [], []
            for i in range(sequence_length, len(scaled_data) - future_target):
                X_train.append(scaled_data[i-sequence_length:i])
                y_train.append(scaled_data[i + future_target, 0])  # Target is the scaled close price after 14 days

            X_train, y_train = np.array(X_train), np.array(y_train)
            model = build_and_train_model(X_train, y_train)

            # Predict the 14-day ahead price
            last_60_days = scaled_data[-sequence_length:]
            last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))
            future_price_prediction = model.predict(last_60_days)
            future_price_prediction = scaler.inverse_transform([[future_price_prediction[0][0]] + [0]*(scaled_data.shape[1]-1)])[0][0]

            # Get the last actual close price before prediction
            last_close_price = data['Close'].iloc[-1]
            percentage_increase = ((future_price_prediction - last_close_price) / last_close_price) * 100
            
            # Display results
            st.success(f"The last day price is {last_close_price:.2f}, the expected price is {future_price_prediction:.2f}, "
                       f"with an increase of {percentage_increase:.2f}%")

if __name__ == "__main__":
    main()
