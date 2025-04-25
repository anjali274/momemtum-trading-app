@st.cache_data
def get_stock_data(ticker, start_date="2020-01-01"):
    try:
        # Fetch stock data using yfinance
        df = yf.Ticker(ticker).history(start=start_date)
        
        # Ensure all characters are properly encoded
        df = df.applymap(lambda x: str(x) if isinstance(x, str) else x)
        
        df["20EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["50EMA"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["RSI"] = compute_rsi(df["Close"])
        df["MACD"], df["Signal"] = compute_macd(df["Close"])
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error
