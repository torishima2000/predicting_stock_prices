# 取得データの比較6

# 自作プログラム
# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mylibrary as mylib

# 複数銘柄の取得
Security_codes = ["7203.T", "9984.T", "6861.T"]
my_hists = []

for t in Security_codes:
    mylib.stock_prices_to_csv(t)
    df = mylib.get_stock_prices(t)
    df = df[df.index >= "2021-04-26"]
    my_hists.append(df)


# サイトのプログラムのコピー
import yfinance as yf

tickers = yf.Tickers("7203.T 9984.T 6861.T")
hists = []

for i in range(len(tickers.tickers)):
    hists.append(tickers.tickers[i].history())


# 取得したDataFrameオブジェクトの比較
print(my_hists[0] == hists[0])
