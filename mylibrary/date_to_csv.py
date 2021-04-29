# データからcsvファイルを作成するモジュール群

# モジュールのインポート
import os
import pandas as pd
import yfinance as yf

def stock_prices_to_csv(security_code, directory_name = os.path.join("HistoricalDate", "StockPriceValues"), file_name = None):
    """価格のヒストリカルデータを取得し、csvファイルに記憶する

    Args:
        security_code (string): 銘柄コード
        directory_name (:obj: string , optional):
            csvファイルを作成するディレクトリへの相対パス
            デフォルトでは、"HistoricalDate\StockPriceValues"
        file_name (:obj: string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker("{}.T".format(security_code))

    # 価格データをDateFrameとして取得する
    hist = ticker.history(period="max")

    # データをcsvファイルで保存する
    os.makedirs(directory_name, exist_ok = True)
    if file_name:
        file_name = file_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(directory_name, file_name)
    hist.to_csv(path, sep = ",")
