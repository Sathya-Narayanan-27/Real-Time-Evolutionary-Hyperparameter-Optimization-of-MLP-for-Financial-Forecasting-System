import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from indicators import add_indicators
from models import evaluate_models, build_mlp
from ga_optimizer import run_ga


st.set_page_config(layout="wide")

st.title("📈 AI Financial Forecasting System")

file = st.file_uploader("Upload Dataset")


if file:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(file)

    df = df.sort_values("Date")

    # ---------------- CLEAN DATA ----------------
    df.replace("-", np.nan, inplace=True)

    numeric_cols = [
        "Open","High","Low","Close",
        "P/E","P/B","Div Yield %"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill()

    # ---------------- ADD TECHNICAL INDICATORS ----------------
    df = add_indicators(df)

    # ---------------- FEATURE SCALING ----------------
    features = [
        "Open","High","Low","Close",
        "P/E","P/B","Div Yield %",
        "RSI","EMA10","EMA20","MACD"
    ]

    scaler = MinMaxScaler()

    df[features] = scaler.fit_transform(df[features])

    # ---------------- CANDLESTICK CHART ----------------
    st.subheader("Candlestick Chart")

    rows = st.slider("Candlestick Data Range", 50, 500, 200)

    chart_data = df.tail(rows)

    fig = go.Figure(data=[go.Candlestick(
        x=chart_data["Date"],
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"]
    )])

    st.plotly_chart(fig, width="stretch")

    # ---------------- SLIDING WINDOW ----------------
    window = 10

    data = df[features].values

    X = []
    y = []

    for i in range(len(data)-window):

        X.append(data[i:i+window])
        y.append(data[i+window][3])

    X = np.array(X)
    y = np.array(y)

    # ---------------- DATA SPLIT ----------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        shuffle=False
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        shuffle=False
    )

    # ---------------- MODEL COMPARISON ----------------
    st.subheader("Model Comparison")

    if "model_results" not in st.session_state:

        results = evaluate_models(
            X_train,
            X_test,
            y_train,
            y_test
        )

        st.session_state.model_results = results

    results = st.session_state.model_results

    table = pd.DataFrame(
        results.items(),
        columns=["Model","RMSE"]
    )

    st.table(table)

    # ---------------- GENETIC OPTIMIZATION ----------------
    run_ga_button = st.button("Run Genetic Optimization")

    if run_ga_button or "ga_solution" in st.session_state:

        if "ga_solution" not in st.session_state:

            with st.spinner("Running Genetic Algorithm Optimization..."):

                solution, fitness_history = run_ga(
                    X_train,
                    y_train,
                    X_val,
                    y_val
                )

                st.session_state.ga_solution = solution
                st.session_state.ga_fitness = fitness_history


        solution = st.session_state.ga_solution
        fitness_history = st.session_state.ga_fitness


        n1 = int(solution[0])
        n2 = int(solution[1])
        lr = solution[2]
        batch = int(solution[3])


        model = build_mlp(
            X_train.shape[1:],
            n1,
            n2,
            lr
        )

        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=batch,
            verbose=0
        )

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # ---------------- UPDATED MODEL TABLE ----------------
        updated_results = results.copy()

        updated_results["GA Optimized MLP"] = rmse

        updated_table = pd.DataFrame(
            updated_results.items(),
            columns=["Model","RMSE"]
        )

        st.subheader("Updated Model Comparison")

        st.table(updated_table)

        # ---------------- GA FITNESS GRAPH ----------------
        st.subheader("GA Fitness Evolution")

        st.line_chart(fitness_history)

        # ---------------- ACTUAL VS PREDICTED ----------------
        st.subheader("Actual vs Predicted")

        chart = pd.DataFrame({
            "Actual": y_test,
            "Predicted": preds.flatten()
        })

        st.line_chart(chart)

        # ---------------- NEXT DAY PREDICTION ----------------
        next_pred = model.predict(
            X_test[-1].reshape(1, *X_test[-1].shape),
            verbose=0
        )

        pred_array = np.zeros((1, len(features)))
        pred_array[0][3] = next_pred[0][0]

        actual_price = scaler.inverse_transform(pred_array)[0][3]

        # ---------------- CONFIDENCE LEVEL ----------------
        confidence = max(0, (1 - rmse) * 100)

        st.subheader("Next Day Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Close Price",
                value=f"₹ {actual_price:.2f}"
            )

        with col2:
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

        # ---------------- TREND INDICATOR ----------------
        current_price = df["Close"].iloc[-1]

        trend = "📈 Bullish" if actual_price > current_price else "📉 Bearish"

        st.write(f"### Market Trend: {trend}")

        # ---------------- CONFIDENCE MESSAGE ----------------
        if confidence > 90:
            st.success("High Confidence Prediction")
        elif confidence > 70:
            st.warning("Moderate Confidence Prediction")
        else:
            st.error("Low Confidence Prediction")