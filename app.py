import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator, TSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel

# Streamlit app
st.title("Stock Price Prediction with CNN-LSTM")

# User inputs
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "EGAL.CA")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-10-23"))

if st.button("Run Prediction"):
    # Load data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the specified date range and ticker.")
    else:
        data = data[['Close', 'High', 'Low', 'Volume']]

        # Calculate RSI for different periods
# Calculate RSI for different periods and flatten the result
        data['RSI_3'] = RSIIndicator(data['Close'], window=3).rsi().values.flatten()
        data['RSI_5'] = RSIIndicator(data['Close'], window=5).rsi().values.flatten()
        data['RSI_9'] = RSIIndicator(data['Close'], window=9).rsi().values.flatten()
        data['RSI_14'] = RSIIndicator(data['Close'], window=14).rsi().values.flatten()
        data['RSI_20'] = RSIIndicator(data['Close'], window=20).rsi().values.flatten()


        # Momentum Indicators
        data['TSI'] = TSIIndicator(data['Close']).tsi()
        data['WilliamsR'] = WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
        data['Stochastic'] = StochasticOscillator(close=data['Close'], high=data['High'], low=data['Low']).stoch()

        # Trend Indicators
        data['SMA_10'] = SMAIndicator(data['Close'], window=10).sma_indicator()
        data['SMA_50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
        data['EMA_10'] = EMAIndicator(data['Close'], window=10).ema_indicator()
        data['EMA_50'] = EMAIndicator(data['Close'], window=50).ema_indicator()
        data['MACD'] = MACD(data['Close']).macd_diff()  # MACD Difference
        data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()

        # Volatility Indicators
        bollinger = BollingerBands(close=data['Close'])
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()
        data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        keltner = KeltnerChannel(high=data['High'], low=data['Low'], close=data['Close'])
        data['Keltner_High'] = keltner.keltner_channel_hband()
        data['Keltner_Low'] = keltner.keltner_channel_lband()

        # Volume Indicators
        data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        vwap = VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
        data['VWAP'] = vwap.volume_weighted_average_price()

        # Calculate Volatility as Standard Deviation of Closing Prices over 20 days
        data['Volatility'] = data['Close'].rolling(window=20).std()

        # Drop NaN values caused by indicators with look-back periods
        data = data.dropna()

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare training dataset
        sequence_length = 60
        future_target = 14  # Predict the price 14 days ahead
        X_train, y_train = [], []

        for i in range(sequence_length, len(scaled_data) - future_target):
            X_train.append(scaled_data[i-sequence_length:i])
            y_train.append(scaled_data[i + future_target, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        # Build CNN-LSTM model
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

        # Train model
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Predict the 2-week (14-day) ahead price
        last_60_days = scaled_data[-sequence_length:]
        last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))
        future_price_prediction = model.predict(last_60_days)
        future_price_prediction = scaler.inverse_transform([[future_price_prediction[0][0]] + [0]*(scaled_data.shape[1]-1)])[0][0]

        # Get the last actual close price before prediction
        last_close_price = data['Close'].iloc[-1]
        # Calculate the percentage increase
        percentage_increase = ((future_price_prediction - last_close_price) / last_close_price) * 100

        # Display results
        st.write(f"The last day price is {last_close_price:.2f}")
        st.write(f"The expected price is {future_price_prediction:.2f}")
        st.write(f"Expected increase is {percentage_increase:.2f}%")
