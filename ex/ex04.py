# 計量的・実証的トレーディングの準備
# キャッシュフロー計算書
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E3%82%AD%E3%83%A3%E3%83%83%E3%82%B7%E3%83%A5%E3%83%95%E3%83%AD%E3%83%BC%E8%A8%88%E7%AE%97%E6%9B%B8

import yfinance as yf

ticker = yf.Ticker("7203.T")
cashflow = ticker.cashflow
print(cashflow)