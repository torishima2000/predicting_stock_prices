# 取得データの比較1

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mylibrary as mylib

# 価格のヒストリカルデータ
Security_code = "7203.T"
mylib.stock_prices_to_csv(Security_code)
my_hist = mylib.get_stock_prices(Security_code)


# サイトのプログラムのコピー
import yfinance as yf

ticker = yf.Ticker("7203.T")
hist = ticker.history(period="max")


# 取得したDataFrameオブジェクトの比較
print(my_hist.equals(hist))
