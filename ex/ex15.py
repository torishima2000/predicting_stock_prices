# 計量的・実証的トレーディングの実行
# 終値データフレームの整形、および月次リターンデータフレームの作成
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E7%B5%82%E5%80%A4%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E6%95%B4%E5%BD%A2%E3%81%8A%E3%82%88%E3%81%B3%E6%9C%88%E6%AC%A1%E3%83%AA%E3%82%BF%E3%83%BC%E3%83%B3%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%81%AE%E4%BD%9C%E6%88%90

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

monthly_rt = closes.pct_change().shift(-1)                                # 月次リターンの作成(ラグあり)
monthly_rt = monthly_rt.sub(monthly_rt["^N225"], axis=0)                  # マーケットリターン控除

closes = closes[closes.index > datetime.datetime(2017, 4, 1)]             # 2017年4月以降
monthly_rt = monthly_rt[monthly_rt.index > datetime.datetime(2017, 4, 1)]

closes = closes.drop(["^N225", "month", "end_of_month"], axis=1)          # 不要なカラムを削除
monthly_rt = monthly_rt.drop(["^N225", "month", "end_of_month"], axis=1)

print(closes)
print(monthly_rt)