# モジュールのインポート
import numpy as np
import pandas as pd
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 当期純利益データフレームの作成

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mylib.get_codelist_topix500()

# データの取得
for s in topix500_codes:
    mylib.pl_to_csv(str(s) + ".T")

# 当期純利益
my_earnings = []

# 当期純利益をリストとして記憶
my_dummy = mylib.get_pl(str(topix500_codes[0]) + ".T")["Net Income"]
my_dummy[:] = np.nan
for s in topix500_codes:
    df = mylib.get_pl(str(s) + ".T")
    try:
        my_earnings.append(df["Net Income"])
    except:
        my_earnings.append(my_dummy)
my_earnings.append(my_dummy)

# 当期純利益のリストをDateFrame化
my_earnings = pd.DataFrame(my_earnings).T
# カラム名の指定
my_earnings.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
my_earnings = my_earnings.sort_index()

print(my_earnings)
