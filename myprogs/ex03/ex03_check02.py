# 取得データの比較3

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 貸借対照表（バランスシート）
Security_code = "7203.T"
mylib.balance_sheet_to_csv(Security_code)
my_balance_sheet = mylib.get_balance_sheet(Security_code).T


# サイトのプログラムのコピー
import yfinance as yf

ticker = yf.Ticker("7203.T")
balance_sheet = ticker.balance_sheet


# 取得したDataFrameオブジェクトの比較
print(my_balance_sheet.columns.dtype == balance_sheet.columns.dtype)
