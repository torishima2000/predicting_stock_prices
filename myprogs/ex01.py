# 自作モジュールのインポート
import mylibrary as mylib

# 価格のヒストリカルデータ
Security_code = "7203.T"
mylib.stock_prices_to_csv(Security_code)
hist = mylib.get_stock_prices(Security_code)
print(hist)
