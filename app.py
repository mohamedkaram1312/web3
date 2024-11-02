import streamlit as st 
import numpy as np 
import pandas as pd 
import yfinance as yf 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout 
from keras.callbacks import EarlyStopping 
from datetime import datetime 

# Function to calculate RSI 
def calculate_rsi(data, window): 
    delta = data['Close'].diff() 
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() 
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean() 
    rs = gain / loss 
    return 100 - (100 / (1 + rs)) 

# Function to calculate %K and %D 
def calculate_stochastic_oscillator(data, k_window=14, d_window=3): 
    low_min = data['Low'].rolling(window=k_window).min() 
    high_max = data['High'].rolling(window=k_window).max() 
    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min)) 
    data['%D'] = data['%K'].rolling(window=d_window).mean() 
    return data 

# Function to create sequences for LSTM 
def create_sequences(data, step): 
    X, y = [], [] 
    for i in range(len(data) - step): 
        X.append(data[i:i + step]) 
        y.append(data[i + step, 0])  # Target is Close price 
    return np.array(X), np.array(y) 

# Function to analyze stock and predict next 30 days prices using multiple indicators 
def analyze_stock(ticker, start_date, end_date): 
    data = yf.download(ticker, start=start_date, end=end_date) 
    
    # Calculate indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean() 
    data['SMA_200'] = data['Close'].rolling(window=200).mean() 
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean() 
    data['RSI_3'] = calculate_rsi(data, 3) 
    data['RSI_5'] = calculate_rsi(data, 5) 
    data['RSI_10'] = calculate_rsi(data, 10) 
    data['RSI_14'] = calculate_rsi(data, 14) 
    data['RSI_20'] = calculate_rsi(data, 20) 
    data = calculate_stochastic_oscillator(data) 

    # Drop rows with NaN values 
    data = data.dropna() 

    # Prepare data for scaling (select features) 
    scaler = MinMaxScaler() 
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_50', 'SMA_200', 'EMA_50', 
                                             'RSI_3', 'RSI_5', 'RSI_10', 'RSI_14', 'RSI_20', 
                                             '%K', '%D']]) 

    # Create sequences for LSTM model 
    X, y = create_sequences(scaled_data, step=5) 

    # Train/test split 
    split = int(len(X) * 0.8) 
    X_train, X_test = X[:split], X[split:] 
    y_train, y_test = y[:split], y[split:] 

    # Build LSTM model 
    model = Sequential([ 
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])), 
        Dropout(0.2), 
        LSTM(128, return_sequences=False), 
        Dropout(0.2), 
        Dense(1) 
    ]) 
    model.compile(optimizer='adam', loss='mean_squared_error') 

    # Early stopping to prevent overfitting 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 

    # Train the model 
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0) 

    # Predict the next 30 days prices
    last_sequence = scaled_data[-60:] 
    predictions = [] 
    for _ in range(30): 
        last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1])) 
        predicted_price_scaled = model.predict(last_sequence) 
        
        # Invert scaling to get the predicted price in original units 
        predicted_price = scaler.inverse_transform(
            np.array([[predicted_price_scaled[0][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        )[0][0] 
        predictions.append(predicted_price) 
        
        # Update the last_sequence with the new prediction 
        new_sequence = np.append(last_sequence[0][1:], [[predicted_price_scaled[0][0]] + [0]*10], axis=0) 
        last_sequence = new_sequence 

    return data['Close'].iloc[-1].item(), predictions

# Streamlit app 
st.title("30-Day Stock Price Prediction") 

# User input for ticker and date range 
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):") 
start_date = st.date_input("Select Start Date", datetime(2021, 1, 1)) 
end_date = st.date_input("Select End Date", datetime(2024, 1, 1)) 

if st.button("Predict"): 
    if ticker and start_date < end_date: 
        with st.spinner(f"Fetching data for {ticker} from {start_date} to {end_date}..."): 
            last_price, predictions = analyze_stock(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) 

            # Display prediction results 
            st.write("### Prediction") 
            st.write(f"**Last Close Price**: ${last_price:.2f}")
            st.write("**30-Day Predicted Prices:**")
            for i, price in enumerate(predictions, start=1):
                st.write(f"Day {i}: ${price:.2f}")
    else: 
        st.error("Please enter a valid ticker and ensure the end date is after the start date.")
