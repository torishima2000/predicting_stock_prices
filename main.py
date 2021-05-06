# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自作モジュールのインポート
import mylibrary as mine

# main

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mine.get_codelist_topix500()

# 終値データフレームの作成
# 終値
closes = []
# 終値をリストとして記憶
for s in topix500_codes:
    df = mine.get_stock_prices(str(s) + ".T")
    closes.append(df.Close)

df = mine.get_stock_prices("^N225")
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
dummy = mine.get_pl(str(topix500_codes[0]) + ".T")["Net Income"]
dummy[:] = np.nan
for s in topix500_codes:
    df = mine.get_pl(str(s) + ".T")
    try:
        earnings.append(df["Net Income"])
    except:
        earnings.append(dummy)
earnings.append(dummy)

# 終値のリストをDateFrame化
earnings = pd.DataFrame(earnings).T
# カラム名の指定
earnings.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
earnings = earnings.sort_index()
