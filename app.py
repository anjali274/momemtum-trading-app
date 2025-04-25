# ─── Comparison of Previous vs Today ─────────────────────────────────────────────

st.divider()
st.subheader("📊 Compare Previous Day vs Today")

# Let the user select the stock and dates for comparison
ticker_to_compare = st.text_input("Enter Stock Ticker for Comparison (e.g., MSFT):")

# Use two date pickers for selecting comparison dates
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

        # Plot recent prices with EMAs
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close")
        ax.plot(df.index, df["20EMA"], label="20EMA", linestyle="--")
        ax.plot(df.index, df["50EMA"], label="50EMA", linestyle="--")
        ax.set_title(f"{ticker_to_compare.upper()} Price & EMAs")
        ax.legend()
        st.pyplot(fig)

        # RSI and MACD
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
