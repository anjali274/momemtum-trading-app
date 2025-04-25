import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ─── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_data
def load_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table["Symbol"].tolist(), table.set_index("Symbol")["Security"].to_dict()

@st.cache_data
def get_stock_data(ticker, start_date="2020-01-01", end_date=None):
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
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
    reason = []
    
    cond1 = latest["20EMA"] > latest["50EMA"]
    cond2 = 50 <= latest["RSI"] <= 70
    cond3 = latest["MACD"] > latest["Signal"]

    if cond1:
        reason.append("20EMA is above 50EMA")
    else:
        reason.append("20EMA is not above 50EMA")

    if cond2:
        reason.append("RSI is in the optimal range (50-70)")
    else:
        reason.append("RSI is outside optimal range (50-70)")

    if cond3:
        reason.append("MACD is above Signal")
    else:
        reason.append("MACD is not above Signal")

    is_buy = cond1 and cond2 and cond3
    return is_buy, "; ".join(reason)

# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("🔍 Momentum Screener: Top-N Stocks")

# 1. Select how many top stocks to show
top_n = st.selectbox("How many top momentum stocks?", options=[5, 10, 20, 50, 100], index=2)

# 2. Button to run full screen
if st.button("🏁 Run Screener"):
    tickers, ticker_names = load_sp500_tickers()
    results = []

    with st.spinner("Fetching data and screening..."):
        for sym in tickers:
            try:
                df = get_stock_data(sym, start_date="2023-01-01")
                if len(df) < 50:
                    continue
                is_buy, reason = check_buy_signal(df)
                if is_buy:
                    one_month_ago = df["Close"].iloc[-21]
                    latest_price = df["Close"].iloc[-1]
                    ret = (latest_price / one_month_ago) - 1
                    results.append((sym, ticker_names.get(sym, sym), ret, reason, df))
            except Exception as e:
                print(f"Couldn't fetch {sym}: {e}")

    res_df = pd.DataFrame(
        [(s, n, r, rsn) for s, n, r, rsn, _ in results],
        columns=["Ticker", "Name", "1-Mo Return", "Reason"]
    ).sort_values("1-Mo Return", ascending=False).reset_index(drop=True)

    st.subheader(f"Top {top_n} Momentum Stocks")
    top_df = res_df.head(top_n)
    st.dataframe(top_df.style.format({"1-Mo Return": "{:.2%}"}))

    sel = st.selectbox("Pick a ticker for detailed view", options=top_df["Ticker"].tolist())
    if sel:
        df_sel = next(df for s, _, _, _, df in results if s == sel)
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

# ─── Comparison of Previous vs Today ─────────────────────────────────────────────

st.divider()
st.subheader("📊 Compare Previous Day vs Today")

ticker_to_compare = st.text_input("Enter Stock Ticker for Comparison (e.g., MSFT):")
today_date = st.date_input("Select Today's Date", datetime.today())
previous_date = st.date_input("Select Previous Date", datetime.today() - timedelta(days=1))

if ticker_to_compare and st.button("Compare Stocks"):
    df = get_stock_data(ticker_to_compare.upper(), start_date=previous_date - timedelta(days=50), end_date=today_date)

    if df.empty or len(df) < 30:
        st.error("Couldn't fetch sufficient data for the selected dates.")
    else:
        st.markdown(f"### 📈 {ticker_to_compare.upper()} Analysis from {previous_date} to {today_date}")
        latest = df.loc[df.index <= pd.to_datetime(today_date)].iloc[-1]
        st.markdown(f"**Latest Close Price**: {latest['Close']:.2f}")

        buy_signal, reason = check_buy_signal(df)
        st.markdown(f"### 🟢 Buy Signal: `{buy_signal}`")
        st.markdown(f"**Reason:** {reason}")

        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df["20EMA"], label="20EMA", linestyle="--")
        ax.plot(df.index, df["50EMA"], label="50EMA", linestyle="--")
        ax.set_title(f"{ticker_to_compare.upper()} Price & EMAs")
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
