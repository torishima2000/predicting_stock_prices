# 【第二回】機械学習で株価予測（TA-LibとLightGBMを使った学習モデル構築）
# TA-Libで株価のテクニカル指標をチェック

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import mylibrary as mylib


# 株価データの取得
df = mylib.get_stock_prices("7203.T")

# 元データの抽出
begin = datetime.datetime(*[2018, 5, 16])
end = datetime.datetime(*[2020, 1, 26])
df = df[df.index >= begin]
df = df[df.index <= end]


# 移動平均線の算出
df["SMA5"] = talib.SMA(np.array(df["Close"]), timeperiod=5)
df["SMA25"] = talib.SMA(np.array(df["Close"]), timeperiod=25)
df["SMA50"] = talib.SMA(np.array(df["Close"]), timeperiod=50)
df["SMA75"] = talib.SMA(np.array(df["Close"]), timeperiod=75)
df["SMA100"] = talib.SMA(np.array(df["Close"]), timeperiod=100)

# 移動平均線の描画
sma = { "SMA5":df["SMA5"],
        "SMA25":df["SMA25"],
        "SMA50":df["SMA50"],
        "SMA75":df["SMA75"],
        "SMA100":df["SMA100"]}
mylib.plot_chart(sma)


# ボリンジャーバンドの算出
df["upper1"], middle, df["lower1"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
df["upper2"], middle, df["lower2"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)
df["upper3"], middle, df["lower3"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=3, nbdevdn=3, matype=0)

# ボリンジャーバンドの描画
bbands = {  "upper1":df["upper1"],
            "lower1":df["lower1"],
            "upper2":df["upper2"],
            "lower2":df["lower2"],
            "upper3":df["upper3"],
            "lower3":df["lower3"],}
mylib.plot_chart(bbands)


# MACDの算出
df["MACD"], df["MACDsignal"], df["MACDhist"] = talib.MACD(np.array(df["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)

# MACDの描画
macd = {"MACD":df["MACD"],
        "MACDsignal":df["MACDsignal"],
        "MACDhist":df["MACDhist"]}
mylib.plot_chart(macd)


# RSIの算出
df["RSI9"] = talib.RSI(np.array(df["Close"]), timeperiod=9)
df["RSI14"] = talib.RSI(np.array(df["Close"]), timeperiod=14)

# RSIの描画
rsi = { "RSI9":df["RSI9"],
        "RSI14":df["RSI14"]}
mylib.plot_chart(rsi)
