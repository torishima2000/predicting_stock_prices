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
    os.makedirs(settings.directory_names["stock_prices"], exist_ok = True)
    if file_name:
        file_name = file_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_names["stock_prices"], file_name)
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
    os.makedirs(settings.directory_names["Profit_and_Loss_Statement"], exist_ok = True)
    if file_name:
        file_name = fine_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_names["Profit_and_Loss_Statement"], file_name)
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
    os.makedirs(settings.directory_names["balance_sheet"], exist_ok = True)
    if file_name:
        file_name = fine_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_names["balance_sheet"], file_name)
    balance_sheet.to_csv(path, sep = ",")

def cash_flow_statement_to_csv(security_code, file_name = None):
    """過去3年分のキャッシュ・フロー計算書を取得し、csvファイルに記憶する
    保存先は .\HistoricalDate\CashFlowStatement

    Args:
        security_code (string): 銘柄コード
        file_name (string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker("{}.T".format(security_code))

    # キャッシュ・フロー計算書をDateFrameオブジェクトで取得
    cashflow = ticker.cashflow

    # データをcsvファイルで保存する
    os.makedirs(settings.directory_names["cash_flow_statement"], exist_ok = True)
    if file_name:
        file_name = fine_name + ".csv"
    else:
        file_name = security_code + ".csv"
    path = os.path.join(settings.directory_names["cash_flow_statement"], file_name)
    cashflow.to_csv(path, sep = ",")

def topix500_to_csv(file_name = "TOPIX500"):
    """TOPIX500構成銘柄の取得

    Args:
        file_name (str, optional): 
            保存するcsvファイル名.
            デフォルトは "TOPIX500".
    """
    # 東証上場銘柄一覧を取得
    path_to_jp = os.path.join(settings.directory_names["TSE_listed_Issues"],
                              settings.file_names["TSE_listed_Issues_JP"])
    path_to_en = os.path.join(settings.directory_names["TSE_listed_Issues"],
                              settings.file_names["TSE_listed_Issues_EN"])
    issues_jp = pd.read_excel(path_to_jp)
    issues_en = pd.read_excel(path_to_en)

    # データの形成
    # TOPIX500構成銘柄の行だけ摘出
    issues = issues_en[(issues_en["Size (New Index Series)"] == "TOPIX Core30") |
                       (issues_en["Size (New Index Series)"] == "TOPIX Large70") |
                       (issues_en["Size (New Index Series)"] == "TOPIX Mid400")]
    # 銘柄の日本語表記を取得
    name_japanese = issues_jp["銘柄名"]
    # 銘柄の日本語表記を挿入
    issues.insert(3, "Name (Japanese)", name_japanese)

    # データの保存
    issues.to_csv(os.path.join(settings.directory_names["TSE_listed_Issues"], file_name + ".csv"), sep = ",")
