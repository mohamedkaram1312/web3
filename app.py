import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Streamlit configuration and button
st.title("Stock Price Prediction")
st.write("This app allows you to input a stock ticker and date range to predict the next month's price.")

# Get user inputs
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-10-24"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-20"))

if st.button("Run Prediction"):

    def compute_indicators(data):
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # Compute RSI for multiple periods
        for period in [3, 5, 10, 14, 20]:
            delta = data['Close'].diff(1)
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Compute MACD
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Compute OBV
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        
        # Compute CMF
        data['CMF'] = (((data['Close'] - data['Low']) - (data['High'] - data['Close'])) /
                       (data['High'] - data['Low']) * data['Volume']).rolling(window=20).mean()

        # Calculate L14 and H14
        data['L14'] = data['Low'].rolling(window=14, min_periods=1).min()
        data['H14'] = data['High'].rolling(window=14, min_periods=1).max()
        
        # Calculate %K and handle potential division by zero
        data['%K'] = np.where(
            (data['H14'] - data['L14']) == 0,
            np.nan,  # Assign NaN where division is not possible
            100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
        )
        data['%D'] = data['%K'].rolling(window=3).mean()

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
        data['Support'] = data['Low'].rolling(window=20).min()
        data['Resistance'] = data['High'].rolling(window=20).max()
        
        return data

    # Load data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data available for the selected ticker and date range.")
    else:
        data = compute_indicators(data)
        data.dropna(inplace=True)  # Remove any remaining NaN rows

        # Prepare features and target for LSTM
        feature_columns = ['Close', 'SMA_50', 'EMA_50', 'RSI_14', 'MACD', 'Signal', 'OBV', '%K', '%D']
        features = data[feature_columns].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)

        # Split data into train and test sets
        train_size = int(len(features_scaled) * 0.8)
        train, test = features_scaled[:train_size], features_scaled[train_size:]

        # Function to create dataset for LSTM
        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                a = data[i:(i + time_step), :]
                X.append(a)
                y.append(data[i + time_step, 0])  # Predicting 'Close'
            return np.array(X), np.array(y)

        time_step = 30
        X_train, y_train = create_dataset(train, time_step)
        X_test, y_test = create_dataset(test, time_step)

        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer

        # Compile and fit the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], features_scaled.shape[1] - 1))), axis=1))[:, 0]

        # Output predictions
        st.write("Predicted Prices:")
        st.write(pd.DataFrame(y_pred, columns=['Predicted Close']))
