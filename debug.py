# モジュールのインポート
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 自作モジュールのインポート
import mylibrary as mylib

# 銘柄コードを指定
# Security_code = "7203.T"

# 価格のヒストリカルデータを取得し、csvファイルに保存
# mylib.stock_prices_to_csv(Security_code)

# 損益計算書をcsvファイルに保存
# mylib.pl_to_csv(Security_code)

# 貸借対照表をcsvファイルに保存
# mylib.balance_sheet_to_csv(Security_code)

# キャッシュ・フロー計算書をcsvファイルに保存
# mylib.cash_flow_statement_to_csv(Security_code)

# 銘柄のサマリーをcsvファイルに保存
# mylib.sammary_to_csv(Security_code)

# TOPIX100構成銘柄をcsvファイルに保存
# mylib.topix100_to_csv()

# TOPIX500構成銘柄をcsvファイルに保存
# mylib.topix500_to_csv()

# TOPIX500構成銘柄の証券コードを取得
# topix500_codes = mylib.get_codelist_topix500()

# S&P500構成銘柄の証券コードを取得
sandp500_codes = mylib.get_codelist_sandp500()

# 実行済み
# for s in topix500_codes:
#     mylib.stock_prices_to_csv(str(s) + ".T")
# mylib.stock_prices_to_csv("^N225")
# for s in topix500_codes:
#     mylib.pl_to_csv(str(s) + ".T")
# for s in topix500_codes:
#     mylib.balance_sheet_to_csv(str(s) + ".T")
# for s in topix500_codes:
#     mylib.cash_flow_statement_to_csv(str(s) + ".T")
# for s in topix500_codes[150:]:
#     mylib.sammary_to_csv(str(s) + ".T")

bins = [(-0.5 + (i / 25.0)) for i in range(26)]
print(bins)