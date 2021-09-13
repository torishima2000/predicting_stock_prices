# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 株価以外のデータ取得（為替）
import pandas as pd
import yfinance as yf

fxs = ["JPY=X", "EURUSD=X", "GBPUSD=X"]

my_closes = []
for t in fxs:
    mylib.stock_prices_to_csv(t)
    my_closes.append(mylib.get_stock_prices(t).Close)

my_df = pd.DataFrame(my_closes).T
my_df.columns = fxs

print(my_df)
