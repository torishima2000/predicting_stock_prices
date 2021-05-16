# 計量的・実証的トレーディングの準備
# 損益計算書
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E6%90%8D%E7%9B%8A%E8%A8%88%E7%AE%97%E6%9B%B8

import yfinance as yf

ticker = yf.Ticker("7203.T")
financials = ticker.financials
print(financials)