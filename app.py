import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.metrics import mean_squared_error
from src.preprocess import preprocess_data
from src.model import train_lstm_model
from src.predict import predict_next_day, predict_test_set

# Fungsi ambil data BTC sampai hari ini
@st.cache_data
def load_latest_btc_data():
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download("BTC-USD", start="2022-01-01", end=today, progress=False)
    data = data[['Close']].dropna()
    return data

# Fungsi ambil kurs real-time
def get_usd_to_idr_rate():
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()["rates"]["IDR"]
    except Exception as e:
        st.warning(f"Gagal ambil kurs real-time: {e}. Menggunakan kurs tetap Rp15.000.")
        return 15000

# Judul
st.title("📊 Prediksi Harga Bitcoin (BTC) dan Konversi ke Rupiah")

# Tombol refresh
if st.button("🔁 Refresh Prediksi"):
    st.rerun()

# Ambil data terbaru & preprocessing
data = load_latest_btc_data()
X_train, X_test, y_train, y_test, scaler, X_full = preprocess_data(data)

# Latih ulang model langsung saat ini
with st.spinner("🔄 Melatih ulang model berdasarkan data terbaru..."):
    model = train_lstm_model(X_train, y_train)

# Harga live dari yfinance
latest_actual_price = yf.Ticker("BTC-USD").info["regularMarketPrice"]

# Prediksi besok & konversi
next_price_usd = predict_next_day(model, X_full, scaler)
usd_to_idr = get_usd_to_idr_rate()
next_price_idr = next_price_usd * usd_to_idr

# Tampilkan harga dan hasil
st.markdown(f"### 📌 Harga BTC Saat Ini (Live): **${latest_actual_price:,.2f}**")
st.markdown(f"### 💰 Prediksi Harga BTC Besok: **${next_price_usd:,.2f}**")
st.markdown(f"### 🇮🇩 Konversi ke Rupiah: **Rp{next_price_idr:,.0f}**")
st.markdown(f"📌 Kurs Hari Ini: **1 USD = Rp{usd_to_idr:,.2f}**")

# Visualisasi prediksi vs data asli
actual, predicted = predict_test_set(model, X_test, y_test, scaler)
rmse = np.sqrt(mean_squared_error(actual, predicted))
test_dates = data.index[-len(actual):]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_dates, actual, label='Harga Asli (USD)', color='blue')
ax.plot(test_dates, predicted, label='Prediksi (USD)', color='orange')
ax.scatter(test_dates[-1] + pd.Timedelta(days=1), next_price_usd, color='red', label='Prediksi Besok')

ax.set_title(f"Perbandingan Harga BTC Asli vs Prediksi (RMSE: {rmse:.2f})")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (USD)")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# Saran beli berdasarkan perbandingan live vs prediksi
delta = next_price_usd - latest_actual_price
if delta > 0:
    st.success("📈 Prediksi naik → Anda bisa pertimbangkan untuk beli!")
else:
    st.error("📉 Prediksi turun → Mungkin lebih baik menunggu.")

# Footer
st.caption("Data live dari yfinance • Kurs dari open.er-api.com • Model dilatih ulang setiap refresh")