# 取得データの比較4

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# キャッシュフロー計算書
Security_code = "7203.T"
mylib.cash_flow_statement_to_csv(Security_code)
my_cashflow = mylib.get_cash_flow_statement(Security_code).T


# サイトのプログラムのコピー
import yfinance as yf

ticker = yf.Ticker("7203.T")
cashflow = ticker.cashflow

# 取得したDataFrameオブジェクトの比較
print(my_cashflow.index.dtype == cashflow.index.dtype)
print(my_cashflow.columns.dtype == cashflow.columns.dtype)
