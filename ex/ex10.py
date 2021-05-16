# 計量的・実証的トレーディングの実行
# 終値データフレームの作成
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E7%B5%82%E5%80%A4%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E4%BD%9C%E6%88%90

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))

closes   = [] # 終値

for i in range(len(tickers.tickers)):
    closes.append(tickers.tickers[i].history(period="max").Close)

closes = pd.DataFrame(closes).T   # DataFrame化
closes.columns = stocks           # カラム名の設定
closes = closes.ffill()           # 欠損データの補完

print(closes)