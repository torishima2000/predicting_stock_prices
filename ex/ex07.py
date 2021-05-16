# 計量的・実証的トレーディングの準備
# 株価以外のデータ取得（為替）
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E6%A0%AA%E4%BE%A1%E4%BB%A5%E5%A4%96%E3%81%AE%E3%83%87%E3%83%BC%E3%82%BF%E5%8F%96%E5%BE%97%E7%82%BA%E6%9B%BF

import pandas as pd
import yfinance as yf

fxs = ["JPY=X", "EURUSD=X", "GBPUSD=X"]
tickers = yf.Tickers(" ".join(fxs))

closes = []
for i in range(len(tickers.tickers)):
    closes.append(tickers.tickers[i].history(period="max").Close)

df = pd.DataFrame(closes).T
df.columns = fxs

print(df)