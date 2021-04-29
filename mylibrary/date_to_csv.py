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

    # 価格データをDateFrameオブジェクトで取得
    hist = ticker.history(period="max")

    # データをcsvファイルで保存する
    os.makedirs(settings.directory_name["stock_prices"], exist_ok = True)
    if file_name:
        file_name = file_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_name["stock_prices"], file_name)
    hist.to_csv(path, sep = ",")

def pl_to_csv(security_code, file_name = None):
    """過去3年分の損益計算書を取得し、csvファイルに記憶する
    保存先は .\HistoricalDate\Profit_and_Loss_Statement

    Args:
        security_code (string): 銘柄コード
        file_name (:obj: string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker("{}.T".format(security_code))

    # 損益計算書をDateFrameオブジェクトで取得
    financials = ticker.financials

    # データをcsvファイルで保存する
    os.makedirs(settings.directory_name["Profit_and_Loss_Statement"], exist_ok = True)
    if file_name:
        file_name = fine_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_name["Profit_and_Loss_Statement"], file_name)
    financials.to_csv(path, sep = ",")

def balance_sheet_to_csv(security_code, file_name = None):
    """過去3年分の貸借対照表を取得し、csvファイルに記憶する
    保存先は .\HistoricalDate\BalanceSheet

    Args:
        security_code (string): 銘柄コード
        file_name (:obj: string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker("{}.T".format(security_code))

    # 貸借対照表をDateFrameオブジェクトで取得
    balance_sheet = ticker.balance_sheet

    # データをcsvファイルで保存する
    os.makedirs(settings.directory_name["balance_sheet"], exist_ok = True)
    if file_name:
        file_name = fine_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_name["balance_sheet"], file_name)
    balance_sheet.to_csv(path, sep = ",")

def cash_flow_statement_to_csv(security_code, file_name = None):
    """過去3年分のキャッシュ・フロー計算書を取得し、csvファイルに記憶する
    保存先は .\HistoricalDate\CashFlowStatement

    Args:
        security_code (string): 銘柄コード
        file_name (:obj: string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker("{}.T".format(security_code))

    # キャッシュ・フロー計算書をDateFrameオブジェクトで取得
    cashflow = ticker.cashflow

    # データをcsvファイルで保存する
    os.makedirs(settings.directory_name["cash_flow_statement"], exist_ok = True)
    if file_name:
        file_name = fine_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_name["cash_flow_statement"], file_name)
    cashflow.to_csv(path, sep = ",")
