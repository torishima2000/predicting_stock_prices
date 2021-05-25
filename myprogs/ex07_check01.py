# 取得データの比較7

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mylibrary as mylib

# 株価以外のデータ取得（為替）
import pandas as pd

my_fxs = ["JPY=X", "EURUSD=X", "GBPUSD=X"]

my_closes = []
for t in my_fxs:
    mylib.stock_prices_to_csv(t)
    my_closes.append(mylib.get_stock_prices(t).Close)

my_df = pd.DataFrame(my_closes).T
my_df.columns = my_fxs


# サイトのプログラムのコピー
import pandas as pd
import yfinance as yf

fxs = ["JPY=X", "EURUSD=X", "GBPUSD=X"]
tickers = yf.Tickers(" ".join(fxs))

closes = []
for i in range(len(tickers.tickers)):
    closes.append(tickers.tickers[i].history(period="max").Close)

df = pd.DataFrame(closes).T
df.columns = fxs


# 取得したDataFrameオブジェクトの比較
print(my_df.equals(df))
