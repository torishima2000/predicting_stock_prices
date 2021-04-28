# 価格のヒストリカルデータを取得する

# モジュールのインポート
import os
import pandas as pd
import yfinance as yf

# 証券コードを指定
Security_code = "7203"

# ティッカーシンボルを作成
ticker = yf.Ticker("{}.T".format(Security_code))

# 価格データをDateFrameとして取得する
hist = ticker.history(period="max")

# データの保存先のファイルへの相対パス
file_path = os.path.join("historys", "{}.csv".format(Security_code))

# csvファイルとしてデータを保存
hist.to_csv(file_path, sep = ",")
