# 指定データを取得するモジュール群

# モジュールのインポート
import os
import json
import numpy as np
import pandas as pd

# 設定ファイルの読み込み
path_name = {}
with open("mylibrary\path.json", "r") as f:
    path_name = json.load(f)

def get_stock_prices(security_code):
    """価格のヒストリカルデータを取得

    Args:
        security_code (string): 銘柄コード

    Returns:
        [DateFrame]: 価格のヒストリカルデータ
    """
    # 価格のヒストリカルデータをcsvファイルから取得
    path = os.path.join(path_name["stock_prices"], security_code + ".csv")
    stock_prices = pd.read_csv(path, index_col = 0)
    stock_prices.index = pd.to_datetime(stock_prices.index)
    return stock_prices

def get_pl(security_code):
    """過去3年分の損益計算書を取得

    Args:
        security_code (string): 銘柄コード

    Returns:
        [DateFrame]: 過去3年分の損益計算書
    """
    # 損益計算書をcsvファイルから取得
    path = os.path.join(path_name["Profit_and_Loss_Statement"], security_code + ".csv")
    profit_and_loss_statement = pd.read_csv(path, index_col = 0)
    profit_and_loss_statement.index = pd.to_datetime(profit_and_loss_statement.index)
    return profit_and_loss_statement

def get_balance_sheet(security_code):
    """過去3年分の貸借対照表を取得

    Args:
        security_code (string): 銘柄コード

    Returns:
        [DateFrame]: 過去3年分の貸借対照表
    """
    # 貸借対照表をcsvファイルから取得
    path = os.path.join(path_name["balance_sheet"], security_code + ".csv")
    balance_sheet = pd.read_csv(path, index_col = 0)
    balance_sheet.index = pd.to_datetime(balance_sheet.index)
    return balance_sheet

def get_cash_flow_statement(security_code):
    """過去3年分のキャッシュ・フロー計算書を取得

    Args:
        security_code (string): 銘柄コード

    Returns:
        [DateFrame]: 過去3年分のキャッシュ・フロー計算書
    """
    # キャッシュ・フロー計算書をcsvファイルから取得
    path = os.path.join(path_name["cash_flow_statement"], security_code + ".csv")
    cash_flow_statement = pd.read_csv(path, index_col = 0)
    cash_flow_statement.index = pd.to_datetime(cash_flow_statement.index)
    return cash_flow_statement

def get_sammary(security_code):
    """銘柄のサマリーを取得

    Args:
        security_code (string): 銘柄コード

    Returns:
        [Dictionary]: 銘柄のサマリー
    """
    sammary = {}
    path = os.path.join(path_name["sammary"], security_code + ".json")
    with open(path, "r") as f:
        sammary = json.load(f)
    return sammary

def get_codelist_topix100():
    """TOPIX100構成銘柄の証券コードリストを取得

    Returns:
        [list]: TOPIX100構成銘柄の証券コード
    """
    # TOPIX100構成銘柄情報をcsvファイルから取得
    path = os.path.join(path_name["TSE_listed_Issues"], path_name["TOPIX100"] + ".csv")
    list_topix100 = pd.read_csv(path)

    # 証券コード部分のみ摘出
    codes = list_topix100["Local Code"].values.tolist()
    for i, code in enumerate(codes):
        codes[i] = str(code) + ".T"
    return codes

def get_codelist_topix500():
    """TOPIX500構成銘柄の証券コードリストを取得

    Returns:
        [list]: TOPIX500構成銘柄の証券コード
    """
    # TOPIX500構成銘柄情報をcsvファイルから取得
    path = os.path.join(path_name["TSE_listed_Issues"], path_name["TOPIX500"] + ".csv")
    list_topix500 = pd.read_csv(path)

    # 証券コード部分のみ摘出
    codes = list_topix500["Local Code"].values.tolist()
    for i, code in enumerate(codes):
        codes[i] = str(code) + ".T"
    return codes

def get_codelist_sandp500():
    """S&P500構成銘柄の証券コードリストを取得

    Returns:
        [list]: S&P500構成銘柄の証券コード
    """
    # S&P500構成銘柄情報をcsvファイルから取得
    path = os.path.join(path_name["S&P500_components"], path_name["S&P500"] + ".csv")
    list_sandp500 = pd.read_csv(path)

    # 証券コード部分のみ摘出
    codes = list_sandp500["Symbol"].values.tolist()
    return codes

def get_stock_prices_dataframe(tickers, ohlc):
    """OHLCいずれかのデータフレームを取得

    Args:
        tickers [list]: 取得したい銘柄の証券コードのリスト
        ohlc [String]: Open, High, Low, Closeのうちいずれかを指定

    Returns:
        [DataFrame]: 指定したOHLCのデータフレーム
    """
    # データフレームを格納する変数
    ohlc_df = []
    # OHLCの情報をリストとして記憶
    for ticker in tickers:
        df = get_stock_prices(ticker)
        ohlc_df.append(df[ohlc])
    # OHLCのリストをDataFrame化
    ohlc_df = pd.DataFrame(ohlc_df).T
    # カラム名の指定
    ohlc_df.columns = tickers
    # データのソート
    ohlc_df = ohlc_df.sort_index()
    # 欠損データの補完
    ohlc_df = ohlc_df.ffill()
    return ohlc_df

def get_earnings_dataframe(tickers):
    # 当期純利益
    earnings = []
    # 当期純利益をリストとして記憶
    dummy = get_pl("AAPL")["Net Income"]
    dummy[:] = np.nan
    for ticker in tickers:
        df = get_pl(ticker)
        try:
            earnings.append(df["Net Income"])
        except:
            earnings.append(dummy)
    # 当期純利益のリストをDateFrame化
    earnings = pd.DataFrame(earnings).T
    # カラム名の指定
    earnings.columns = tickers
    # データのソート
    earnings = earnings.sort_index()
    return earnings
