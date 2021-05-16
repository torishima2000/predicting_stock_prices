# 計量的・実証的トレーディングの準備
# 銘柄のサマリー
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E9%8A%98%E6%9F%84%E3%81%AE%E3%82%B5%E3%83%9E%E3%83%AA%E3%83%BC

import yfinance as yf

ticker = yf.Ticker("7203.T")
info = ticker.info
print(info)