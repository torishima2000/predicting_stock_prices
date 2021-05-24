# 計量的・実証的トレーディングの準備
# 価格のヒストリカルデータ
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E4%BE%A1%E6%A0%BC%E3%81%AE%E3%83%92%E3%82%B9%E3%83%88%E3%83%AA%E3%82%AB%E3%83%AB%E3%83%87%E3%83%BC%E3%82%BF

import yfinance as yf

ticker = yf.Ticker("7203.T")
hist = ticker.history(period="max")
hist = hist = hist[hist.index <= "2020-11-10"]
print(hist)
