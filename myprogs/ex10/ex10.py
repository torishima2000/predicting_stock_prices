# モジュールのインポート
import pandas as pd
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mylibrary as mylib

# 終値データフレームの作成

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mylib.get_codelist_topix500()

# データの取得
for s in topix500_codes:
    mylib.stock_prices_to_csv(str(s) + ".T")
mylib.stock_prices_to_csv("^N225")

# 終値
my_closes = []
# 終値をリストとして記憶
for s in topix500_codes:
    my_df = mylib.get_stock_prices(str(s) + ".T")
    my_closes.append(my_df.Close)

my_df = mylib.get_stock_prices("^N225")
my_closes.append(my_df.Close)

# 終値のリストをDateFrame化
my_closes = pd.DataFrame(my_closes).T
# カラム名の指定
my_closes.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# データのソート
my_closes = my_closes.sort_index()
# 欠損データの補完
my_closes = my_closes.ffill()
print(my_closes)