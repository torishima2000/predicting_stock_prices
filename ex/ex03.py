# 計量的・実証的トレーディングの準備
# 貸借対照表（バランスシート）
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E8%B2%B8%E5%80%9F%E5%AF%BE%E7%85%A7%E8%A1%A8%E3%83%90%E3%83%A9%E3%83%B3%E3%82%B9%E3%82%B7%E3%83%BC%E3%83%88

import yfinance as yf

ticker = yf.Ticker("7203.T")
balance_sheet = ticker.balance_sheet
print(balance_sheet)