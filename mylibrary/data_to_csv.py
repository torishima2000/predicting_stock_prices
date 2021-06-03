# データからcsvファイルを作成するモジュール群

# モジュールのインポート
import os
import json
import pandas as pd
import yfinance as yf

# 設定ファイルの読み込み
path_name = {}
with open("mylibrary\path.json", "r") as f:
    path_name = json.load(f)

def stock_prices_to_csv(security_code):
    """価格のヒストリカルデータを取得し、csvファイルに記憶する
    保存先は \Dates\HistoricalDate\StockPrices

    Args:
        security_code (string): 銘柄コード
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker(security_code)

    # 価格データをDateFrameオブジェクトで取得
    hist = ticker.history(period="max")

    # データをcsvファイルで保存する
    os.makedirs(path_name["stock_prices"], exist_ok = True)
    file_name = os.path.join(path_name["stock_prices"], security_code + ".csv")
    hist.to_csv(file_name, sep = ",")

def pl_to_csv(security_code):
    """過去3年分の損益計算書を取得し、csvファイルに記憶する
    保存先は \Dates\HistoricalDate\Profit_and_Loss_Statement

    Args:
        security_code (string): 銘柄コード
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker(security_code)

    # 損益計算書をDateFrameオブジェクトで取得
    financials = ticker.financials.T

    # データをcsvファイルで保存する
    os.makedirs(path_name["Profit_and_Loss_Statement"], exist_ok = True)
    file_name = os.path.join(path_name["Profit_and_Loss_Statement"], security_code + ".csv")
    financials.to_csv(file_name, sep = ",")

def balance_sheet_to_csv(security_code):
    """過去3年分の貸借対照表を取得し、csvファイルに記憶する
    保存先は \Dates\HistoricalDate\BalanceSheet

    Args:
        security_code (string): 銘柄コード
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker(security_code)

    # 貸借対照表をDateFrameオブジェクトで取得
    balance_sheet = ticker.balance_sheet.T

    # データをcsvファイルで保存する
    os.makedirs(path_name["balance_sheet"], exist_ok = True)
    file_name = os.path.join(path_name["balance_sheet"], security_code + ".csv")
    balance_sheet.to_csv(file_name, sep = ",")

def cash_flow_statement_to_csv(security_code):
    """過去3年分のキャッシュ・フロー計算書を取得し、csvファイルに記憶する
    保存先は \Dates\HistoricalDate\CashFlowStatement

    Args:
        security_code (string): 銘柄コード
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker(security_code)

    # キャッシュ・フロー計算書をDateFrameオブジェクトで取得
    cashflow = ticker.cashflow.T

    # データをcsvファイルで保存する
    os.makedirs(path_name["cash_flow_statement"], exist_ok = True)
    file_name = os.path.join(path_name["cash_flow_statement"], security_code + ".csv")
    cashflow.to_csv(file_name, sep = ",")

def sammary_to_csv(security_code):
    """銘柄のサマリーを取得し、jsonファイルに記憶する
    保存先は \Dates\Sammary

    Args:
        security_code (string): 銘柄コード
    """
    # ティッカーシンボルを作成
    ticker = yf.Ticker(security_code)

    # 銘柄のサマリーをDateFrameオブジェクトで取得
    info = ticker.info
    
    # データをjsonファイルで保存する
    os.makedirs(path_name["sammary"], exist_ok = True)
    file_name = os.path.join(path_name["sammary"], security_code + ".json")
    with open(file_name, "w") as f:
        json.dump(info, f)


def topix100_to_csv():
    """TOPIX100構成銘柄の取得
    保存先は \Dates\List_of_TSE-listedIssues\[取得したリストの更新年月日]
    """
    # 東証上場銘柄一覧を取得
    path_name_to_jp = os.path.join(path_name["TSE_listed_Issues"],
                              path_name["TSE_listed_Issues_JP"])
    path_name_to_en = os.path.join(path_name["TSE_listed_Issues"],
                              path_name["TSE_listed_Issues_EN"])
    issues_jp = pd.read_excel(path_name_to_jp)
    issues_en = pd.read_excel(path_name_to_en)

    # データの形成
    # TOPIX100構成銘柄の行だけ摘出
    issues = issues_en[(issues_en["Size (New Index Series)"] == "TOPIX Core30") |
                       (issues_en["Size (New Index Series)"] == "TOPIX Large70")]
    # 銘柄の日本語表記を取得
    name_japanese = issues_jp["銘柄名"]
    # 銘柄の日本語表記を挿入
    issues.insert(3, "Name (Japanese)", name_japanese)

    # データの保存
    file_name = path_name["TOPIX100"] + ".csv"
    issues.to_csv(os.path.join(path_name["TSE_listed_Issues"], file_name), sep = ",")


def topix500_to_csv():
    """TOPIX500構成銘柄の取得
    保存先は \Dates\List_of_TSE-listedIssues\[取得したリストの更新年月日]
    """
    # 東証上場銘柄一覧を取得
    path_name_to_jp = os.path.join(path_name["TSE_listed_Issues"],
                              path_name["TSE_listed_Issues_JP"])
    path_name_to_en = os.path.join(path_name["TSE_listed_Issues"],
                              path_name["TSE_listed_Issues_EN"])
    issues_jp = pd.read_excel(path_name_to_jp)
    issues_en = pd.read_excel(path_name_to_en)

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
    file_name = path_name["TOPIX500"] + ".csv"
    issues.to_csv(os.path.join(path_name["TSE_listed_Issues"], file_name), sep = ",")
