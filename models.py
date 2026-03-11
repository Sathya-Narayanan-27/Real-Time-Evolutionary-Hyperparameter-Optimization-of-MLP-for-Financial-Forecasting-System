from pyexpat import model

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input


def build_mlp(input_shape, n1=64, n2=32, lr=0.001):

    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(n1, activation="relu"),
        Dense(n2, activation="relu"),
        Dense(1)
    ])

    from tensorflow.keras.optimizers import Adam

    model.compile(
    optimizer=Adam(learning_rate=lr),
    loss="mse"
)

    return model


def evaluate_models(X_train, X_test, y_train, y_test):

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train.reshape(len(X_train), -1), y_train)

    preds = lr.predict(X_test.reshape(len(X_test), -1))

    results["Linear Regression"] = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10
    )

    rf.fit(X_train.reshape(len(X_train), -1), y_train)

    preds = rf.predict(X_test.reshape(len(X_test), -1))

    results["Random Forest"] = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    # Basic MLP
    mlp = build_mlp(X_train.shape[1:])

    mlp.fit(X_train, y_train, epochs=10, verbose=0)

    preds = mlp.predict(X_test)

    results["Basic MLP"] = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    return results