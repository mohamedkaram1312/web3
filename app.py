import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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
        
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['CMF'] = (((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / 
                    (data['High'] - data['Low']) * data['Volume']).rolling(window=20).mean()
    
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
    
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Resistance'] = data['High'].rolling(window=20).max()
    
    return data

def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])
    return np.array(X), np.array(y)

def analyze_stock(ticker):
    data = yf.download(ticker, start="2022-10-24", end="2024-11-01")
    data = compute_indicators(data)
    
    data.dropna(inplace=True)
    
    if len(data) < 10:
        print(f"Not enough data for {ticker}.")
        return ticker, None, None, None
    
    features = data[['Close', 'SMA_50', 'EMA_50', 'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', 
                     'MACD', 'Signal', 'OBV', 'CMF', '%K', '%D', 'BB_Middle', 'BB_Upper', 
                     'BB_Lower', 'Support', 'Resistance']]
    target = data['Close'].shift(-1)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.values[:-1].reshape(-1, 1))
    
    X, y = create_sequences(features_scaled, step=10)
    y = target_scaled[:len(y)]
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    last_sequence = features_scaled[-10:]
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))
    
    future_price = model.predict(last_sequence)
    predicted_price = scaler_y.inverse_transform(future_price)[0][0]
    last_actual_price = data['Close'].iloc[-1]
    
    percentage_increase = ((predicted_price - last_actual_price) / last_actual_price) * 100
    
    return ticker, data['Close'].iloc[-1], predicted_price, percentage_increase

def main():
    st.title("Stock Price Prediction")
    tickers = ['SUGR.CA']
    results = []
    
    for ticker in tickers:
        ticker, last_price, predicted_price, percentage_increase = analyze_stock(ticker)
        if percentage_increase is not None:
            results.append((ticker, last_price, predicted_price, percentage_increase))
    
    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
    
    for ticker, last_price, predicted_price, percentage_increase in sorted_results:
        st.write(f"{ticker}: Last Price: {last_price:.2f}, Predicted Price: {predicted_price:.2f}, Expected Increase: {percentage_increase:.2f}%")

if __name__ == "__main__":
    main()
