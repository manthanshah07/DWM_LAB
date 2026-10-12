import yfinance as yf
import pandas as pd


def generate_stock_data():
    sectors = {
        "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
        "Auto": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
        "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    }

    # Fallback mapping for tickers renamed on Yahoo Finance (e.g. TATAMOTORS.NS -> TMPV.NS)
    fallback_tickers = {
        "TATAMOTORS.NS": "TMPV.NS",
    }

    all_records = []

    print("Downloading 3 years of daily trading data for 20 major Indian stocks...")
    for sector, tickers in sectors.items():
        for symbol in tickers:
            print(f"Fetching {symbol} ({sector})...")
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(period="3y", interval="1d")

            if df.empty and symbol in fallback_tickers:
                alt_symbol = fallback_tickers[symbol]
                print(
                    f"  {symbol} renamed on Yahoo Finance; fetching via {alt_symbol}..."
                )
                df = yf.Ticker(alt_symbol).history(period="3y", interval="1d")

            if df.empty:
                print(f"Warning: No data returned for {symbol}")
                continue

            df = df.reset_index()

            # Ensure required columns exist and drop nulls before calculations
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[required_cols].dropna()

            clean_ticker = symbol.replace(".NS", "")

            out_df = pd.DataFrame()
            out_df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            out_df["Ticker"] = clean_ticker
            out_df["Sector"] = sector
            out_df["Exchange"] = "NSE"
            out_df["Open_Price"] = df["Open"].round(2)
            out_df["High_Price"] = df["High"].round(2)
            out_df["Low_Price"] = df["Low"].round(2)
            out_df["Close_Price"] = df["Close"].round(2)
            out_df["Volume"] = df["Volume"].astype(int)
            out_df["Daily_Return"] = (
                ((out_df["Close_Price"] - out_df["Open_Price"]) / out_df["Open_Price"])
                * 100
            ).round(2)

            all_records.append(out_df)

    final_df = pd.concat(all_records, ignore_index=True)

    # Ensure no null values exist in the final dataset
    final_df = final_df.dropna()
    final_df = final_df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    output_file = "stock_market_data.csv"
    final_df.to_csv(output_file, index=False)

    print("\nData successfully saved to", output_file)
    print(f"Null values count:\n{final_df.isnull().sum()}\n")
    print(f"Shape of {output_file}: {final_df.shape}")
    print("\nFirst 5 records:")
    print(final_df.head(5).to_string(index=False))


if __name__ == "__main__":
    generate_stock_data()
