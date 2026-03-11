AI Financial Forecasting System using MLP and Genetic Algorithm

An AI-powered financial forecasting system that predicts the next-day stock closing price using historical market data and technical indicators.
The project combines Multilayer Perceptron (MLP) neural networks with Genetic Algorithm (GA) optimization to automatically tune hyperparameters and improve prediction performance.

The system includes an interactive Streamlit dashboard for visualizing financial data, comparing machine learning models, and displaying prediction insights.

Project Overview

Financial markets generate large volumes of time-series data. Traditional statistical models often struggle to capture nonlinear relationships present in stock price movements.

This project uses machine learning and evolutionary optimization techniques to build a predictive model capable of forecasting stock prices based on:

Historical market data

Financial ratios

Technical indicators

The system integrates data preprocessing, feature engineering, neural network training, and optimization into a single interactive platform.

Features

Stock price prediction using MLP Neural Networks

Genetic Algorithm-based hyperparameter optimization

Technical indicator integration

RSI (Relative Strength Index)

EMA (Exponential Moving Average)

MACD (Moving Average Convergence Divergence)

Model comparison between:

Linear Regression

Random Forest

Basic MLP

GA Optimized MLP

Interactive candlestick chart visualization

Genetic Algorithm optimization progress tracking

Next-day stock price prediction

Prediction confidence score

Market trend detection (Bullish / Bearish)

User-friendly Streamlit dashboard

Technologies Used
Technology	Purpose
Python	Core programming language
TensorFlow / Keras	Neural network implementation
Scikit-learn	Machine learning models & evaluation
Streamlit	Interactive dashboard
Plotly	Financial chart visualization
PyGAD	Genetic Algorithm optimization
Pandas	Data processing
NumPy	Numerical computation
TA Library	Technical indicators
