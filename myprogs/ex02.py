# 自作モジュールのインポート
import mylibrary as mylib

# 損益計算書
Security_code = "7203.T"
mylib.pl_to_csv(Security_code)
financials = mylib.get_pl(Security_code)
print(financials.T)
