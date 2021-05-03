# モジュールのインポート
import mylibrary as mine

# 銘柄コードを指定
Security_code = "7203"

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
topix500_codes = mine.codelist_topix500()
