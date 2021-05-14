# 資料作成用のプログラム


# 銘柄リストの読み込み
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")
print(data)


# ティッカーの設定
stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))


# 終値データフレームの作成
closes   = [] # 終値

for i in range(len(tickers.tickers)):
    closes.append(tickers.tickers[i].history(period="max").Close)

closes = pd.DataFrame(closes).T   # DataFrame化
closes.columns = stocks           # カラム名の設定
closes = closes.ffill()           # 欠損データの補完

closes = closes.query('Date <= "2020-11-10"')   # データ範囲の指定

print(closes)
