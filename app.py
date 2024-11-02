import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Function to compute technical indicators
def compute_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    # Stochastic Oscillator
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Drop NaN values
    data.dropna(inplace=True)
    
    return data

# Function to predict price direction
def predict_price_direction(ticker):
    data = yf.download(ticker, start="2022-01-01", end="2024-11-01")
    data = compute_indicators(data)

    # Define features and target
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1 if price increases, else 0
    features = data[['SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD', '%K', '%D']]
    target = data['Target']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Print accuracy and classification report
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(classification_report(y_test, predictions))

    return model, features

# Streamlit app setup
st.title("Stock Price Direction Prediction")

# User input for the ticker
ticker = st.text_input("Enter Ticker Symbol:", "AAPL")

if st.button("Predict Direction"):
    with st.spinner("Analyzing..."):
        model, features = predict_price_direction(ticker)
        # Get the last data point's indicators for prediction
        last_indicators = features.iloc[-1].values.reshape(1, -1)
        direction = model.predict(last_indicators)

        if direction[0] == 1:
            st.success(f"The predicted direction for {ticker} is an **increase** in price.")
        else:
            st.error(f"The predicted direction for {ticker} is a **decrease** in price.")
