import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch stock data
def get_stock_data(ticker, start_date="2020-01-01"):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date)
    df["20EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["50EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    return df

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute MACD
def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Buy signal function
def check_buy_signal(df):
    latest = df.iloc[-1]
    if latest["20EMA"] > latest["50EMA"] and latest["RSI"] >= 50 and latest["RSI"] <= 70 and latest["MACD"] > latest["Signal"]:
        return True
    return False

# Streamlit UI
st.title("Momentum Trading Stock Screener")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").upper()
if st.button("Analyze"):
    df = get_stock_data(ticker)
    st.subheader(f"Stock Data for {ticker}")

    # Plot price with EMAs
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close Price", color="blue")
    ax.plot(df.index, df["20EMA"], label="20-day EMA", color="green", linestyle="--")
    ax.plot(df.index, df["50EMA"], label="50-day EMA", color="red", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Show latest metrics
    latest = df.iloc[-1]
    st.write(f"**Latest Price:** ${latest['Close']:.2f}")
    st.write(f"**20-day EMA:** ${latest['20EMA']:.2f}")
    st.write(f"**50-day EMA:** ${latest['50EMA']:.2f}")
    st.write(f"**RSI:** {latest['RSI']:.2f}")
    st.write(f"**MACD:** {latest['MACD']:.2f}")
    st.write(f"**Signal Line:** {latest['Signal']:.2f}")

    # Check buy signal
    if check_buy_signal(df):
        st.success(f"✅ {ticker} meets the buy criteria!")
    else:
        st.warning(f"⚠️ {ticker} does not meet the buy criteria.")
