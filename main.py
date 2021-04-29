# モジュールのインポート
import mylibrary as mine

# 銘柄コードを指定
Security_code = "7203"

# 価格のヒストリカルデータを取得し、csvファイルに保存
mine.stock_prices_to_csv("7203")
