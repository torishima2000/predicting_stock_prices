# モジュールのインポート
import numpy as np
import pandas as pd
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 自己資本データフレームの作成

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mylib.get_codelist_topix500()

# データの取得
for s in topix500_codes:
    mylib.balance_sheet_to_csv(str(s) + ".T")

# 自己資本
my_equity = []

# 自己資本をリストとして記憶
my_dummy = mylib.get_balance_sheet(str(topix500_codes[0]) + ".T")["Total Stockholder Equity"]
my_dummy[:] = np.nan
for s in topix500_codes:
    df = mylib.get_balance_sheet(str(s) + ".T")
    try:
        my_equity.append(df["Total Stockholder Equity"])
    except:
        my_equity.append(my_dummy)
my_equity.append(my_dummy)

# 自己資本のリストをDateFrame化
my_equity = pd.DataFrame(my_equity).T
# カラム名の指定
my_equity.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
my_equity = my_equity.sort_index()

print(my_equity)
