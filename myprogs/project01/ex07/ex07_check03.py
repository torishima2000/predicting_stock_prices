# 取得データの比較7

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
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
# 誤差の排除 ["JPY=X"]カラム
error_correction = lambda x:round(x, 2)
my_df["JPY=X"] = my_df["JPY=X"].map(error_correction)
df["JPY=X"] = df["JPY=X"].map(error_correction)
# 誤差の排除 ["EURUSD=X"]カラム
error_correction = lambda x:round(x, 12)
my_df["EURUSD=X"] = my_df["EURUSD=X"].map(error_correction)
df["EURUSD=X"] = df["EURUSD=X"].map(error_correction)
# 誤差の排除 ["GBPUSD=X"]カラム
error_correction = lambda x:round(x, 13)
my_df["GBPUSD=X"] = my_df["GBPUSD=X"].map(error_correction)
df["GBPUSD=X"] = df["GBPUSD=X"].map(error_correction)

print(my_df.index.dtype == df.index.dtype)
print(my_df.columns.dtype == df.columns.dtype)
