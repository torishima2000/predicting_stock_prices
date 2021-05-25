# 自作プログラム
# 自作モジュールのインポート
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
import pandas as pd

# 誤差の排除
error_correction = lambda x:round(x, 6)
my_hist = my_hist.applymap(error_correction)
hist = hist.applymap(error_correction)

print(my_hist.equals(hist))
