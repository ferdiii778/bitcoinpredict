import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data_loader import load_btc_data
from src.preprocess import preprocess_data
from src.model import train_lstm_model, save_model, load_existing_model
from src.predict import predict_next_day, predict_test_set

# 🔒 Set seed agar hasil konsisten
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Load data
data = load_btc_data()
print(data.head())  # Debug

# 2. Preprocess
X_train, X_test, y_train, y_test, scaler, X_full = preprocess_data(data)

# 3. Load model atau latih jika belum ada
model_path = "models/btc_lstm_model.h5"
if os.path.exists(model_path):
    print("📦 Memuat model dari file...")
    model = load_existing_model(model_path)
else:
    model = train_lstm_model(X_train, y_train)
    save_model(model, model_path)

# 4. Prediksi hari berikutnya
next_price = predict_next_day(model, X_full, scaler)
print(f"\n📈 Prediksi harga BTC besok: ${next_price:,.2f}")

# 5. Visualisasi hasil test
actual, predicted = predict_test_set(model, X_test, y_test, scaler)
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Ambil tanggal dari data
test_dates = data.index[-len(actual):]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual, label='Harga Asli', color='blue')
plt.plot(test_dates, predicted, label='Prediksi', color='orange')

# Titik prediksi untuk besok
next_date = test_dates[-1] + pd.Timedelta(days=1)
plt.scatter(next_date, next_price, color='red', label='Prediksi Besok')

plt.title(f"Perbandingan Harga BTC Asli vs Prediksi (RMSE: {rmse:.2f})")
plt.xlabel("Tanggal")
plt.ylabel("Harga (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Simpan ke file
plt.savefig("plot.png")
plt.show()
