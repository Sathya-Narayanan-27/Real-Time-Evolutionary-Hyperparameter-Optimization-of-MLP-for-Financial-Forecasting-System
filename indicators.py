import ta

def add_indicators(df):

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    df["EMA10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()

    df = df.dropna()

    return df