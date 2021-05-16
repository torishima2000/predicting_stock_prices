# 計量的・実証的トレーディングの実行
# 自己資本データフレームの作成
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E8%87%AA%E5%B7%B1%E8%B3%87%E6%9C%AC%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E4%BD%9C%E6%88%90

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))

equity   = [] # 自己資本

dummy = tickers.tickers[0].balance_sheet.T["Total Stockholder Equity"]
dummy[:] = np.nan

for i in range(len(tickers.tickers)):
    try:
        equity.append(tickers.tickers[i].balance_sheet.T["Total Stockholder Equity"])
    except:
        equity.append(dummy)         # エラー発生時はダミーを入れる

equity = pd.DataFrame(equity).T      # DataFrame化
equity.columns = stocks              # カラム名の設定

print(equity)