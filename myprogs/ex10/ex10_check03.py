# 取得データの比較10

# 自作プログラム
# モジュールのインポート
import pandas as pd
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 終値データフレームの作成

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mylib.get_codelist_topix500()

# データの取得
for s in topix500_codes:
    mylib.stock_prices_to_csv(str(s) + ".T")
mylib.stock_prices_to_csv("^N225")

# 終値
my_closes = []
# 終値をリストとして記憶
for s in topix500_codes:
    my_df = mylib.get_stock_prices(str(s) + ".T")
    my_closes.append(my_df.Close)

my_df = mylib.get_stock_prices("^N225")
my_closes.append(my_df.Close)

# 終値のリストをDateFrame化
my_closes = pd.DataFrame(my_closes).T
# カラム名の指定
my_closes.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
my_closes = my_closes.sort_index()
# 欠損データの補完
my_closes = my_closes.ffill()


# サイトのプログラムのコピー
import datetime
import numpy as np
# 既にインポート済みなため、コメントアウト
# import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))

closes   = [] # 終値

for i in range(len(tickers.tickers)):
    closes.append(tickers.tickers[i].history(period="max").Close)

closes = pd.DataFrame(closes).T   # DataFrame化
closes.columns = stocks           # カラム名の設定
closes = closes.ffill()           # 欠損データの補完


# 誤差の排除
error_correction = lambda x:round(x, 6)
my_closes = my_closes.applymap(error_correction)
closes = closes.applymap(error_correction)

# データタイプを一致させる
my_closes = my_closes.astype(closes.dtypes)

# 欠損値の補完
my_closes = my_closes.fillna(0)
closes = closes.fillna(0)

# 昨日までのデータ
my_closes = my_closes[my_closes.index < "2021-05-26"]
closes = closes[closes.index < "2021-05-26"]

my_closes = my_closes[my_closes.index > "2017-01-01"]
closes = closes[closes.index > "2017-01-01"]

# 取得したDataFrameオブジェクトの比較
print("DataFrame of my_closes")
print(my_closes)
print("DataFrame of closes")
print(closes)


pd.set_option('display.max_rows', None)
print("compare DataFrames")
print((my_closes == closes).T)
print(my_closes.equals(closes))
