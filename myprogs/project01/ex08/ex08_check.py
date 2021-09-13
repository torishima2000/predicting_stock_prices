# 取得データの比較8

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import mylibrary as mylib

# 銘柄リストの読み込み
my_data = mylib.get_codelist_topix500()


# サイトのプログラムのコピー
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")


# 取得したオブジェクトの比較
# オブジェクトのリスト化
data = data.T.values.tolist()[0]
print(my_data == data)
