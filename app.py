import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ─── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_data
def load_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table["Symbol"].tolist()

@st.cache_data
def get_stock_data(ticker, start_date="2020-01-01"):
    df = yf.Ticker(ticker).history(start=start_date)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    df["20EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["50EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    sig_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig_line

def check_buy_signal(df):
    latest = df.iloc[-1]
    return (
        (latest["20EMA"] > latest["50EMA"]) and
        (50 <= latest["RSI"] <= 70) and
        (latest["MACD"] > latest["Signal"])
    )

# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("📈 Momentum Screener with History & Alerts")

top_n = st.selectbox("How many top momentum stocks?", options=[5, 10, 20, 50, 100], index=2)

if st.button("🏁 Run Screener"):
    with st.spinner("Fetching S&P 500 tickers…"):
        tickers = load_sp500_tickers()

    results = []
    for sym in tickers:
        try:
            df = get_stock_data(sym, start_date="2023-01-01")
            if len(df) < 50:
                continue

            # Filter low-volume stocks
            avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
            if avg_volume < 1_000_000:
                continue

            if check_buy_signal(df):
                one_month_ago = df["Close"].iloc[-21]
                latest_price = df["Close"].iloc[-1]
                ret = (latest_price / one_month_ago) - 1
                vol = df["Close"].iloc[-21:].std()
                score = ret / vol if vol > 0 else 0
                results.append((sym, score, ret, df))
        except Exception:
            continue

    res_df = pd.DataFrame(
        [(s, sc, r) for s, sc, r, _ in results],
        columns=["Ticker", "Momentum Score", "1-Mo Return"]
    ).sort_values("Momentum Score", ascending=False).reset_index(drop=True)

    top_df = res_df.head(top_n)
    st.subheader(f"Top {top_n} Momentum Stocks")
    st.dataframe(top_df.style.format({"Momentum Score": "{:.2f}", "1-Mo Return": "{:.2%}"}))

    # Save for today/yesterday
    today_csv = "top_stocks_today.csv"
    yesterday_csv = "top_stocks_yesterday.csv"

    if os.path.exists(today_csv):
        os.replace(today_csv_
