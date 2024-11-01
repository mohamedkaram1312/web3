import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# List of EGX30 stocks
egx30_tickers = [
    'ABUK.CA', 'COMI.CA', 'CIEB.CA', 'ETEL.CA', 'EFG.CA', 'ESRS.CA', 
    'HRHO.CA', 'MNHD.CA', 'SWDY.CA', 'TALAAT.CA', 'AUTO.CA', 'CCAP.CA', 
    'ORAS.CA', 'JUFO.CA', 'ORWE.CA', 'PHDC.CA', 'PACHIN.CA', 'AMER.CA', 
    'MFPC.CA', 'CLHO.CA', 'ISPH.CA', 'SKPC.CA', 'FWRY.CA', 'DCRC.CA', 
    'TAMWEEL.CA', 'ALCN.CA', 'SUGR.CA', 'EGTS.CA', 'BINV.CA', 'EGCH.CA'
]

# Define indicator calculation functions
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = np.maximum(high.diff(), close.shift() - low.diff())
    tr = np.maximum(tr, low.diff())
    tr = tr.rolling(window=window).sum()
    return tr

def compute_momentum(data, window=14):
    return data['Close'].diff(window)

def compute_tsi(data, window=14):
    price_change = data['Close'].diff()
    ema1 = price_change.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    return ema2 / ema1

def get_recommendation(data):
    signals = []
    for index in range(len(data)):
        rsi = data['RSI'].iloc[index]
        adx = data['ADX'].iloc[index]
        momentum = data['Momentum'].iloc[index]
        tsi = data['TSI'].iloc[index]
        buy_signals = sum([rsi < 30, adx > 25, momentum > 0, tsi > 0])
        sell_signals = sum([rsi > 70, momentum < 0, tsi < 0])
        total_signals = buy_signals + sell_signals
        
        if total_signals > 0:
            if buy_signals / total_signals >= 0.8:
                signals.append("booming buy")
            elif buy_signals > sell_signals:
                signals.append("buy")
            elif sell_signals > buy_signals:
                signals.append("sell")
            else:
                signals.append("hold")
        else:
            signals.append("hold")
        
        if rsi < 30:
            signals[-1] = "oversold"
        elif rsi > 70:
            signals[-1] = "overbought"

    return signals

# Streamlit app
st.title("Stock Analysis Recommendations")
st.write("Your smart assistant for stock market insights and recommendations.")

# One Stock Status
st.header("One Stock Status")
ticker = st.text_input("Ticker (e.g., AAPL)")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
if st.button("Get Recommendation"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data available for the provided ticker.")
    else:
        data['RSI'] = compute_rsi(data['Close'])
        data['ADX'] = compute_adx(data)
        data['Momentum'] = compute_momentum(data)
        data['TSI'] = compute_tsi(data)
        recommendations = get_recommendation(data)
        recommendation = recommendations[-1]
        st.success(f"Recommendation for {ticker}: {recommendation}")

# Summary Indicators EGX30
st.header("Summary Indicators EGX30")
start_date_summary = st.date_input("Start Date (Summary)")
end_date_summary = st.date_input("End Date (Summary)")
if st.button("Get EGX30 Summary"):
    recommendations_summary = {'oversold': [], 'booming buy': [], 'buy': [], 'overbought': []}
    for ticker in egx30_tickers:
        data = yf.download(ticker, start=start_date_summary, end=end_date_summary)
        if data.empty:
            continue
        data['RSI'] = compute_rsi(data['Close'])
        data['ADX'] = compute_adx(data)
        data['Momentum'] = compute_momentum(data)
        data['TSI'] = compute_tsi(data)
        recommendations = get_recommendation(data)
        last_recommendation = recommendations[-1]
        if last_recommendation in recommendations_summary:
            recommendations_summary[last_recommendation].append(ticker)

    summary = {key: (value if value else "None") for key, value in recommendations_summary.items()}
    st.write("Summary of Recommendations:")
    st.write(summary)

# Oversold to Buy/Booming
st.header("Oversold to Buy/Booming")
start_date_booming = st.date_input("Start Date (Booming)")
end_date_booming = st.date_input("End Date (Booming)")
if st.button("Check for Oversold Stocks"):
    booming_buy_stocks = []
    for ticker in egx30_tickers:
        data = yf.download(ticker, start=start_date_booming, end=end_date_booming)
        if data.empty:
            continue
        data['RSI'] = compute_rsi(data['Close'])
        data['ADX'] = compute_adx(data)
        data['Momentum'] = compute_momentum(data)
        data['TSI'] = compute_tsi(data)
        recommendations = get_recommendation(data)
        last_recommendation = recommendations[-1]
        if last_recommendation == "oversold":
            booming_buy_stocks.append(ticker)

    st.write("Booming Buy Stocks:")
    st.write(booming_buy_stocks)

