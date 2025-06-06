import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, window_size=7):
    df = df[['Close']].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler, X