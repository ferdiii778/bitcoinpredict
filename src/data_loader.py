import yfinance as yf
import pandas as pd

def load_btc_data():
    print("📥 Mengambil data dari yfinance...")
    data = yf.download("BTC-USD", start="2022-01-01", end="2024-12-31", progress=False)

    # Ambil hanya kolom Close dan buang semua index tambahan
    df = data.reset_index()[['Date', 'Close']]
    df = df.set_index('Date')
    df = df.dropna()

    print(df.head())  # Debug: pastikan benar
    return df
