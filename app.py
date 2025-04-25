import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# —— Helper Functions ——

@st.cache_data
def load_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table["Symbol"].tolist()

@st.cache_data
def get_stock_data(ticker, start_date="2020-01-01"):
    df = yf.Ticker(ticker).history(start=start_date)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    df["20EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["50EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    return df.dropna()

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
    is_buy = (
        (latest["20EMA"] > latest["50EMA"]) and
        (50 <= latest["RSI"] <= 70) and
        (latest["MACD"] > latest["Signal"])
    )
    reasons = []
    if latest["20EMA"] <= latest["50EMA"]:
        reasons.append("20EMA is not above 50EMA")
    if not (50 <= latest["RSI"] <= 70):
        reasons.append("RSI not in buy range (50-70)")
    if latest["MACD"] <= latest["Signal"]:
        reasons.append("MACD not above signal line")
    return is_buy, reasons

# —— UI ——

st.title("🔍 Momentum Screener: Top-N Stocks")

# Top-N Selector
top_n = st.selectbox("How many top momentum stocks?", options=[5, 10, 20, 50, 100], index=2)

# Screener Run
if st.button("🏋️ Run Screener"):
    with st.spinner("Fetching S&P 500 list…"):
        tickers = load_sp500_tickers()

    results = []
    for sym in tickers:
        try:
            df = get_stock_data(sym, start_date="2023-01-01")
            if len(df) < 50:
                continue
            is_buy, reasons = check_buy_signal(df)
            if is_buy:
                one_month_ago = df["Close"].iloc[-21]
                latest_price = df["Close"].iloc[-1]
                ret = (latest_price / one_month_ago) - 1
                results.append((sym, ret, df))
        except:
            continue

    res_df = pd.DataFrame([(s, r) for s, r, _ in results], columns=["Ticker", "1-Mo Return"])
    res_df = res_df.sort_values("1-Mo Return", ascending=False).reset_index(drop=True)
    st.subheader(f"Top {top_n} Momentum Stocks")
    top_df = res_df.head(top_n)
    st.dataframe(top_df.style.format({"1-Mo Return": "{:.2%}"}))

    sel = st.selectbox("Pick a ticker for detailed view", options=top_df["Ticker"].tolist())
    if sel:
        df_sel = next(df for s, _, df in results if s == sel)
        st.markdown(f"### Detailed View: **{sel}**")

        fig, ax = plt.subplots()
        ax.plot(df_sel.index, df_sel["Close"], label="Close")
        ax.plot(df_sel.index, df_sel["20EMA"], label="20-day EMA", linestyle="--")
        ax.plot(df_sel.index, df_sel["50EMA"], label="50-day EMA", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(df_sel.index, df_sel["RSI"], label="RSI")
        ax1.axhline(70, color="red", linestyle="--")
        ax1.axhline(30, color="green", linestyle="--")
        ax1.legend()

        ax2.plot(df_sel.index, df_sel["MACD"], label="MACD")
        ax2.plot(df_sel.index, df_sel["Signal"], label="Signal")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.legend()
        st.pyplot(fig2)

# Comparison section
st.markdown("---")
st.subheader("📊 Compare Previous Day vs Today")
ticker = st.text_input("Enter Stock Ticker for Comparison (e.g., MSFT):").strip().upper()
today_date = st.date_input("Select Today's Date", datetime.today())
prev_date = st.date_input("Select Previous Date", datetime.today())

if ticker and today_date and prev_date:
    df = get_stock_data(ticker, start_date=str(prev_date))
    if df.empty:
        st.error("No data found for the selected stock.")
    else:
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[~df.index.isna()]
            st.markdown(f"### 📈 {ticker} Analysis from {prev_date} to {today_date}")
            try:
                latest = df.loc[df.index <= pd.to_datetime(today_date)].iloc[-1]
                previous = df.loc[df.index <= pd.to_datetime(prev_date)].iloc[-1]
            except IndexError:
                st.error("No matching data for the selected dates.")
            else:
                is_buy_now, reasons_now = check_buy_signal(df[df.index <= pd.to_datetime(today_date)])
                is_buy_prev, reasons_prev = check_buy_signal(df[df.index <= pd.to_datetime(prev_date)])

                st.write("**Today**: ", "✅ Buy" if is_buy_now else "❌ Not a Buy")
                if not is_buy_now:
                    st.write("Reasons:", ", ".join(reasons_now))

                st.write("**Previous**: ", "✅ Buy" if is_buy_prev else "❌ Not a Buy")
                if not is_buy_prev:
                    st.write("Reasons:", ", ".join(reasons_prev))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
