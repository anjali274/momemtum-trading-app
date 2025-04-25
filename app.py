import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

# ─── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_data
def load_sp500_tickers():
    table = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return table["Symbol"].tolist()

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date="2020-01-01"):
    df = yf.Ticker(ticker).history(start=start_date)
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
    return (
        (latest["20EMA"] > latest["50EMA"]) and
        (50 <= latest["RSI"] <= 70) and
        (latest["MACD"] > latest["Signal"])
    )

def process_ticker(sym):
    try:
        df = get_stock_data(sym, start_date="2023-01-01")
        if len(df) < 50:
            return None
        if check_buy_signal(df):
            one_month_ago = df["Close"].iloc[-21]
            latest_price  = df["Close"].iloc[-1]
            ret           = (latest_price / one_month_ago) - 1
            return (sym, ret, df)
    except Exception:
        return None
    return None

# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("🔍 Momentum Screener: Top-N Stocks")

top_n = st.selectbox(
    "How many top momentum stocks?",
    options=[5, 10, 20, 50, 100],
    index=2,
)

if st.button("🏁 Run Screener"):
    with st.spinner("Fetching S&P 500 list…"):
        tickers = load_sp500_tickers()

    with st.spinner("Processing stock data…"):
        # Optional: limit number of tickers during development
        # tickers = tickers[:100]
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(process_ticker, tickers))

        results = [r for r in results if r]

    res_df = pd.DataFrame(
        [(s, r) for s, r, _ in results],
        columns=["Ticker", "1-Mo Return"]
    ).sort_values("1-Mo Return", ascending=False).reset_index(drop=True)

    st.subheader(f"Top {top_n} Momentum Stocks")
    top_df = res_df.head(top_n)
    st.dataframe(top_df.style.format({"1-Mo Return": "{:.2%}"}))

    sel = st.selectbox(
        "Pick a ticker for detailed view",
        options=top_df["Ticker"].tolist()
    )
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
        ax1.set_ylabel("RSI")
        ax1.legend()

        ax2.plot(df_sel.index, df_sel["MACD"], label="MACD")
        ax2.plot(df_sel.index, df_sel["Signal"], label="Signal")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_ylabel("MACD")
        ax2.legend()

        st.pyplot(fig2)
