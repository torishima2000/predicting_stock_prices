# shaping-with-yfinance
**yfinanceを用いたデータ整形**

参考：https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E9%8A%98%E6%9F%84%E3%81%AE%E3%82%B5%E3%83%9E%E3%83%AA%E3%83%BC

モジュールを使用する場合は、mylibraryをインポートする

モジュール一覧:

stock_prices_to_csv(security_code, file_name = None)
   
    価格のヒストリカルデータを取得し、csvファイルに記憶する
    保存先は .\HistoricalDate\StockPrices
    Args:
        security_code (string): 銘柄コード
        file_name (:obj: string , optional): 
            保存するcsvファイルの名前
            デフォルトでは銘柄コードが使用される


設定ファイルにあるパスとファイル名一覧
# 株価のヒストリカルデータ
"stock_prices":os.path.join("Dates", "HistoricalDate", "StockPrices"), 
# 損益計算書
"Profit_and_Loss_Statement":os.path.join("Dates", "HistoricalDate", "ProfitAndLossStatement"),
# 貸借対照表
"balance_sheet":os.path.join("Dates", "HistoricalDate", "BalanceSheet"),
# キャッシュフロー計算書
"cash_flow_statement":os.path.join("Dates", "HistoricalDate", "CashFlowStatement"),
# サマリー
"sammary":os.path.join("Dates", "Sammary"),
# 東証上場銘柄一覧
"TSE_listed_Issues":os.path.join("Dates", "List_of_TSE-listedIssues", "202103"),
# 東証上場銘柄一覧(日本語)
"TSE_listed_Issues_JP":"data_j.xls",
# 東証上場銘柄一覧(英語)
"TSE_listed_Issues_EN":"data_e.xls",
# TOPIX500構成銘柄一覧
"TOPIX500":"TOPIX500"
