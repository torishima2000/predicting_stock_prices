# 取得データの比較2

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 損益計算書
Security_code = "7203.T"
mylib.pl_to_csv(Security_code)
my_financials = mylib.get_pl(Security_code).T


# サイトのプログラムのコピー
import yfinance as yf

ticker = yf.Ticker("7203.T")
financials = ticker.financials


# 取得したDataFrameオブジェクトの比較
import pandas as pd
import numpy as np

# 欠損値の補完
my_financials = my_financials.fillna(np.nan)
financials = financials.fillna(np.nan)

print(my_financials.equals(financials))
