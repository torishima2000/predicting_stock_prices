# 計量的・実証的トレーディングの実行
# EPS、ROEデータフレームの作成
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#epsroe%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E4%BD%9C%E6%88%90

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))

earnings = [] # 当期純利益

dummy = tickers.tickers[0].financials.T["Net Income"]
dummy[:] = np.nan

for i in range(len(tickers.tickers)):
    try:
        earnings.append(tickers.tickers[i].financials.T["Net Income"])
    except:
        earnings.append(dummy)       # エラー発生時はダミーを入れる

earnings = pd.DataFrame(earnings).T  # DataFrame化
earnings.columns = stocks            # カラム名の設定

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

shares   = [] # 発行株数

for i in range(len(tickers.tickers)):
    try:
        shares.append(tickers.tickers[i].info["sharesOutstanding"])
    except:
        shares.append(np.nan)        # エラー発生時はNAN値を入れる

shares = pd.Series(shares)           # Series化
shares.index = stocks                # インデックス名の設定

eps = earnings/shares.values      # EPS
roe = earnings/equity             # ROE

eps = eps.ffill()                 # 欠損データの補完
roe = roe.ffill()

eps = eps.drop(["^N225"], axis=1) # ^N225カラムは削除しておく
roe = roe.drop(["^N225"], axis=1)

print(eps)
print(roe)