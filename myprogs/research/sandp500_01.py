# S&P500構成銘柄
# 2017/04/01 ~ 2021/03/31
# PER(株価収益率)10倍以下, ROE(自己資本利益率)10%以上

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mylibrary as mylib


# ステータスの設定
# 対象銘柄のリスト
tickers = mylib.get_codelist_sandp500()
# データ取得範囲
begin_date = datetime.datetime(2017, 4, 1)
end_date = datetime.datetime(2021, 3, 31)
# 銘柄選定基準値
per_reference_value = 10
roe_reference_value = 0.1


# データの取得
# 既に取得しているデータ部分はコメントアウト済み
# for ticker in tickers:
#     mylib.stock_prices_to_csv(ticker)
# mylib.stock_prices_to_csv("^GSPC")
# for ticker in tickers:
#     mylib.pl_to_csv(ticker)
for ticker in tickers:
    mylib.balance_sheet_to_csv(ticker)
# for ticker in tickers:
#     mylib.sammary_to_csv(ticker)