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
