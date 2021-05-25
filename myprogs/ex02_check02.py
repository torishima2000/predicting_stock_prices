# 取得データの比較2

# 自作プログラム
# 自作モジュールのインポート
import mylibrary as mylib

# 損益計算書
Security_code = "7203.T"
mylib.pl_to_csv(Security_code)
my_financials = mylib.get_pl(Security_code)


# サイトのプログラムのコピー
import yfinance as yf

ticker = yf.Ticker("7203.T")
financials = ticker.financials


# 取得したDataFrameオブジェクトの比較
import pandas as pd
# 誤差の排除
# error_correction = lambda x:round(x, 6)
# my_financials = my_financials.applymap(error_correction)
# financials = financials.applymap(error_correction)

print(my_financials == financials)
