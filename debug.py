# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自作モジュールのインポート
import mylibrary as mine

# 銘柄コードを指定
Security_code = "7203.T"

# 価格のヒストリカルデータを取得し、csvファイルに保存
# mine.stock_prices_to_csv(Security_code)

# 損益計算書をcsvファイルに保存
# mine.pl_to_csv(Security_code)

# 貸借対照表をcsvファイルに保存
# mine.balance_sheet_to_csv(Security_code)

# キャッシュ・フロー計算書をcsvファイルに保存
# mine.cash_flow_statement_to_csv(Security_code)

# 銘柄のサマリーをcsvファイルに保存
# mine.sammary_to_csv(Security_code)

# TOPIX500構成銘柄をcsvファイルに保存
# mine.topix500_to_csv()

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mine.get_codelist_topix500()


# 実行済み
# for s in topix500_codes:
#     mine.stock_prices_to_csv(str(s) + ".T")
# mine.stock_prices_to_csv("^N225")
# for s in topix500_codes:
#     mine.pl_to_csv(str(s) + ".T")

# main

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
