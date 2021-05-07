# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自作モジュールのインポート
import mylibrary as mylib

# main


# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mylib.get_codelist_topix500()


# 終値データフレームの作成
# 終値
closes = []
# 終値をリストとして記憶
for s in topix500_codes:
    df = mylib.get_stock_prices(str(s) + ".T")
    closes.append(df.Close)

df = mylib.get_stock_prices("^N225")
closes.append(df.Close)

# 終値のリストをDateFrame化
closes = pd.DataFrame(closes).T
# カラム名の指定
closes.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# 欠損データの補完
closes = closes.ffill()


# 当期純利益データフレームの作成
# 当期純利益
earnings = []

# 当期純利益をリストとして記憶
dummy = mylib.get_pl(str(topix500_codes[0]) + ".T")["Net Income"]
dummy[:] = np.nan
for s in topix500_codes:
    df = mylib.get_pl(str(s) + ".T")
    try:
        earnings.append(df["Net Income"])
    except:
        earnings.append(dummy)
earnings.append(dummy)

# 当期純利益のリストをDateFrame化
earnings = pd.DataFrame(earnings).T
# カラム名の指定
earnings.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
earnings = earnings.sort_index()


# 自己資本データフレームの作成
# 自己資本
equity = []

# 自己資本をリストとして記憶
dummy = mylib.get_balance_sheet(str(topix500_codes[0]) + ".T")["Total Stockholder Equity"]
dummy[:] = np.nan
for s in topix500_codes:
    df = mylib.get_balance_sheet(str(s) + ".T")
    try:
        equity.append(df["Total Stockholder Equity"])
    except:
        equity.append(dummy)
equity.append(dummy)

# 自己資本のリストをDateFrame化
equity = pd.DataFrame(equity).T
# カラム名の指定
equity.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
equity = equity.sort_index()

# 発行株数データフレームの作成
# 発行株数
shares = []

for s in topix500_codes:
    df = mylib.get_sammary(str(s) + ".T")
    try:
        shares.append(df["sharesOutstanding"])
    except:
        shares.append(np.nan)
shares.append(np.nan)

# 発行株数のリストをSeries化
shares = pd.Series(shares)
# インデックス名の指定
shares.index = [str(s) + ".T" for s in topix500_codes] + ["^N225"]


# EPS(一株当たり利益), ROE(自己資本利益率)のデータフレームの作成
# EPS(一株当たり利益)
eps = earnings / shares.values
# ROE(自己資本利益率)
roe = earnings / equity

# 欠損データ
eps = eps.ffill()
roe = roe.ffill()

eps = eps.drop(["^N225"], axis = 1)
roe = roe.drop(["^N225"], axis = 1)

print(roe)
