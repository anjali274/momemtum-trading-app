import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ─── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_data
def load_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table[["Symbol", "Security"]].rename(columns={"Symbol": "Ticker", "Security": "Name"})

@st.cache_data
def get_stock_data(ticker, start_date="2020-01-01"):
    df = yf.Ticker(ticker).history(start=start_date)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df["20EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["50EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"]   = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta).clip(lower=0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema  = series.ewm(span=long,  adjust=False).mean()
    macd      = short_ema - long_ema
    sig_line  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig_line

def check_buy_signal(df):
    latest = df.iloc[-1]
    reasons = []
    buy = True

    if latest["20EMA"] <= latest["50EMA"]:
        buy = False
        reasons.append("20EMA is not above 50EMA")
    if not (50 <= latest["RSI"] <= 70):
        buy = False
        reasons.append("RSI not in 50-70 range")
    if latest["MACD"] <= latest["Signal"]:
        buy = False
        reasons.append("MACD not above Signal line")

    if buy:
        reasons.append("Meets all criteria")

    return buy, "; ".join(reasons)

# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("🔍 Momentum Screener: Top-N Stocks")

# Load Tickers with Names
ticker_table = load_sp500_tickers()

# 1. Select how many top stocks to show
top_n = st.selectbox("How many top momentum stocks?", options=[5, 10, 20, 50, 100], index=2)

if st.button("🏁 Run Screener"):
    with st.spinner("Fetching data…"):
        results = []
        for _, row in ticker_table.iterrows():
            sym, name = row["Ticker"], row["Name"]
            df = get_stock_data(sym, start_date="2023-01-01")
            if len(df) < 50:
                continue
            buy, reason = check_buy_signal(df)
            if buy:
                one_month_ago = df["Close"].iloc[-21]
                latest_price  = df["Close"].iloc[-1]
                ret           = (latest_price / one_month_ago) - 1
                results.append((sym, name, ret, df, reason))

    res_df = pd.DataFrame([(s, n, r, reason) for s, n, r, _, reason in results], columns=["Ticker", "Name", "1-Mo Return", "Reason"])
    res_df = res_df.sort_values("1-Mo Return", ascending=False).reset_index(drop=True)

    st.subheader(f"Top {top_n} Momentum Stocks")
    st.dataframe(res_df.head(top_n).style.format({"1-Mo Return": "{:.2%}"}))

    sel = st.selectbox("Pick a ticker for detailed view", options=res_df.head(top_n)["Ticker"].tolist())
    if sel:
        df_sel = next(df for s, _, _, df, _ in results if s == sel)
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
        ax1.set_ylabel("RSI")
        ax1.legend()

        ax2.plot(df_sel.index, df_sel["MACD"], label="MACD")
        ax2.plot(df_sel.index, df_sel["Signal"], label="Signal")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_ylabel("MACD")
        ax2.legend()

        st.pyplot(fig2)

# ─── Individual Stock Check ─────────────────────────────────────────────────────

st.markdown("---")
st.header("🔎 Individual Stock Check")
ticker_input = st.text_input("Enter Stock Ticker (e.g., MSFT):", "").upper()

if ticker_input:
    df = get_stock_data(ticker_input, start_date="2023-01-01")
    if df.empty:
        st.error("No data found for this ticker.")
    else:
        buy, reason = check_buy_signal(df)
        name_row = ticker_table[ticker_table["Ticker"] == ticker_input]
        name = name_row["Name"].values[0] if not name_row.empty else ""
        st.subheader(f"{ticker_input} - {name}")
        st.write(f"**Buy Recommendation:** {'✅ Buy' if buy else '❌ Not a Buy'}")
        st.write(f"**Reason:** {reason}")

        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df["20EMA"], label="20EMA", linestyle="--")
        ax.plot(df.index, df["50EMA"], label="50EMA", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(df.index, df["RSI"], label="RSI")
        ax1.axhline(70, color="red", linestyle="--")
        ax1.axhline(30, color="green", linestyle="--")
        ax1.set_ylabel("RSI")
        ax1.legend()

        ax2.plot(df.index, df["MACD"], label="MACD")
        ax2.plot(df.index, df["Signal"], label="Signal")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_ylabel("MACD")
        ax2.legend()

        st.pyplot(fig2)
