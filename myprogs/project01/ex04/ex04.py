# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# キャッシュフロー計算書
Security_code = "7203.T"
mylib.cash_flow_statement_to_csv(Security_code)
my_cashflow = mylib.get_cash_flow_statement(Security_code).T
print(my_cashflow)
