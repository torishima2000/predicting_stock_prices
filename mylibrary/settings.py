# 設定ファイル

# モジュールのインポート
import os

# 保存先のディレクトリを記憶する辞書
directory_name = {
    # 株価のヒストリカルデータ
    "stock_prices":os.path.join("HistoricalDate", "StockPrices"), 
    # 損益計算書
    "Profit_and_Loss_Statement":os.path.join("HistoricalDate", "ProfitAndLossStatement"),
    # 貸借対照表
    "balance_sheet":os.path.join("HistoricalDate", "BalanceSheet"),
    # キャッシュフロー計算書
    "cash_flow_statement":os.path.join("HistoricalDate", "CashFlowStatement")
    }
