# 自作モジュールのインポート
import mylibrary as mylib

# 貸借対照表（バランスシート）
Security_code = "7203.T"
mylib.balance_sheet_to_csv(Security_code)
my_balance_sheet = mylib.get_balance_sheet(Security_code).T
print(my_balance_sheet)
