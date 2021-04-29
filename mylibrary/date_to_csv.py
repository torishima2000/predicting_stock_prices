# データからcsvファイルを作成するモジュール群

# モジュールのインポート
import os
import pandas as pd
import yfinance as yf

# 設定ファイルのインポート
from . import settings

def stock_prices_to_csv(security_code, file_name = None):
    """価格のヒストリカルデータを取得し、csvファイルに記憶する
    保存先は .\HistoricalDate\StockPrices

    Args:
        security_code (string): 銘柄コード
        file_name (:obj: string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker("{}.T".format(security_code))

    # 価格データをDateFrameとして取得する
    hist = ticker.history(period="max")

    # データをcsvファイルで保存する
    os.makedirs(settings.directory_name["stock_prices"], exist_ok = True)
    if file_name:
        file_name = file_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_name["stock_prices"], file_name)
    hist.to_csv(path, sep = ",")
