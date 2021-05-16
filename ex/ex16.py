# 計量的・実証的トレーディングの実行
# PER、ROEデータフレームの作成（月次リターンと同次元）
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#perroe%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E4%BD%9C%E6%88%90%E6%9C%88%E6%AC%A1%E3%83%AA%E3%82%BF%E3%83%BC%E3%83%B3%E3%81%A8%E5%90%8C%E6%AC%A1%E5%85%83

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

closes["month"] = closes.index.month                                      # 月カラムの作成
closes["end_of_month"] = closes.month.diff().shift(-1)                    # 月末フラグカラムの作成
closes = closes[closes.end_of_month != 0]                                 # 月末のみ抽出

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

monthly_rt = closes.pct_change().shift(-1)                                # 月次リターンの作成(ラグあり)
monthly_rt = monthly_rt.sub(monthly_rt["^N225"], axis=0)                  # マーケットリターン控除

closes = closes[closes.index > datetime.datetime(2017, 4, 1)]             # 2017年4月以降
monthly_rt = monthly_rt[monthly_rt.index > datetime.datetime(2017, 4, 1)]

closes = closes.drop(["^N225", "month", "end_of_month"], axis=1)          # 不要なカラムを削除
monthly_rt = monthly_rt.drop(["^N225", "month", "end_of_month"], axis=1)

eps_df = pd.DataFrame(index=monthly_rt.index, columns=monthly_rt.columns) # 月次リターンと同次元のDF作成
roe_df = pd.DataFrame(index=monthly_rt.index, columns=monthly_rt.columns)

for i in range(len(eps_df)):                                              # 各行への代入
    eps_df.iloc[i] = eps[eps.index < eps_df.index[i]].iloc[-1]

for i in range(len(roe_df)):
    roe_df.iloc[i] = roe[roe.index < roe_df.index[i]].iloc[-1]

per_df = closes/eps_df                                                    # PERデータフレームの作成

print(per_df)
print(roe_df)