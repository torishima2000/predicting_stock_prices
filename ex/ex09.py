# 計量的・実証的トレーディングの実行
# ティッカーの設定
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E3%83%86%E3%82%A3%E3%83%83%E3%82%AB%E3%83%BC%E3%81%AE%E8%A8%AD%E5%AE%9A

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))
print(tickers)