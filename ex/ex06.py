# 計量的・実証的トレーディングの準備
# 複数銘柄の取得
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E8%A4%87%E6%95%B0%E9%8A%98%E6%9F%84%E3%81%AE%E5%8F%96%E5%BE%97

import yfinance as yf

tickers = yf.Tickers("7203.T 9984.T 6861.T")
hists = []

for i in range(len(tickers.tickers)):
    hists.append(tickers.tickers[i].history())

print(hists[0])
