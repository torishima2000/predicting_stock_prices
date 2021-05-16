# 計量的・実証的トレーディングの実行
# 発行株数データフレームの作成
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E7%99%BA%E8%A1%8C%E6%A0%AA%E6%95%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E4%BD%9C%E6%88%90

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))

shares   = [] # 発行株数

for i in range(len(tickers.tickers)):
    try:
        shares.append(tickers.tickers[i].info["sharesOutstanding"])
    except:
        shares.append(np.nan)        # エラー発生時はNAN値を入れる

shares = pd.Series(shares)           # Series化
shares.index = stocks                # インデックス名の設定

print(shares)