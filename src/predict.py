import numpy as np

def predict_next_day(model, X_full, scaler):
    last_input = X_full[-1].reshape(1, X_full.shape[1], 1)
    prediction = model.predict(last_input)
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]

def predict_test_set(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicted_prices = scaler.inverse_transform(predictions)
    return actual_prices.flatten(), predicted_prices.flatten()