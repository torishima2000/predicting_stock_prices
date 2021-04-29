# モジュールのインポート
import my_modules as mine


# 銘柄コードを指定
Security_code = "7203"

# 価格のヒストリカルデータを取得する
mine.stock_prices_to_csv("7203")
