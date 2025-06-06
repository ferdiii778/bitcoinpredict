import tensorflow as tf

def train_lstm_model(X_train, y_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("🚀 Training model...")
    model.fit(X_train, y_train, epochs=25, batch_size=32)
    return model

def save_model(model, path):
    tf.keras.models.save_model(model, path)
    print(f"✅ Model disimpan di {path}")

def load_existing_model(path):
    return tf.keras.models.load_model(path)