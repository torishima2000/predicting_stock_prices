# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# ティッカーの設定
my_tickers = mylib.get_codelist_topix500()
my_tickers.append("^N225")
print(my_tickers)
