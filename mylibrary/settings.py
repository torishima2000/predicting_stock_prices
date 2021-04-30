# 設定ファイル

# モジュールのインポート
import os

# 保存先のディレクトリを記憶する辞書
directory_names = {
    # 株価のヒストリカルデータ
    "stock_prices":os.path.join("HistoricalDate", "StockPrices"), 
    # 損益計算書
    "Profit_and_Loss_Statement":os.path.join("HistoricalDate", "ProfitAndLossStatement"),
    # 貸借対照表
    "balance_sheet":os.path.join("HistoricalDate", "BalanceSheet"),
    # キャッシュフロー計算書
    "cash_flow_statement":os.path.join("HistoricalDate", "CashFlowStatement"),
    # 東証上場銘柄一覧
    "TSE_listed_Issues":os.path.join("List_of_TSE-listedIssues", "202103")
    }

# 保存ファイル名を記憶する辞書
file_names = {
    # 東証上場銘柄一覧(日本語)
    "TSE_listed_Issues_JP":"data_j.xls",
    # 東証上場銘柄一覧(英語)
    "TSE_listed_Issues_EN":"data_e.xls",
    # TOPIX500構成銘柄一覧
    "TOPIX500":"TOPIX500"
}
