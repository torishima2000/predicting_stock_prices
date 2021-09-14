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
plt.figure(figsize=(10.24, 7.68))
plt.plot(df["SMA5"], label="SMA5")
plt.plot(df["SMA25"], label="SMA25")
plt.plot(df["SMA50"], label="SMA50")
plt.plot(df["SMA75"], label="SMA75")
plt.plot(df["SMA100"], label="SMA100")
plt.legend()
plt.xlabel("date")
plt.ylabel("price")
plt.show()
plt.close()


# ボリンジャーバンドの算出
df["upper1"], df["middle"], df["lower1"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
df["upper2"], df["middle"], df["lower2"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)
df["upper3"], df["middle"], df["lower3"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=3, nbdevdn=3, matype=0)

# ボリンジャーバンドの描画
plt.figure(figsize=(10.24, 7.68))
plt.plot(df["upper1"], label="upper1")
plt.plot(df["lower1"], label="lower1")
plt.plot(df["upper2"], label="upper2")
plt.plot(df["lower2"], label="lower2")
plt.plot(df["upper3"], label="upper3")
plt.plot(df["lower3"], label="lower3")
plt.legend()
plt.xlabel("date")
plt.ylabel("price")
plt.show()
plt.close()
