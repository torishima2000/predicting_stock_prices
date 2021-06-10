# TOPIX500構成銘柄
# 2017/04/01 ~ 2021/03/31
# PER(株価収益率)10倍以下, ROE(自己資本利益率)10%以上

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mylibrary as mylib


# ステータスの設定
# 対象銘柄のリスト
tickers = mylib.get_codelist_topix500()
reference_ticker = "^N225"
# データ取得範囲
begin = [2017, 4, 1]
begin_date = datetime.datetime(*begin)
end = [2021, 3, 31]
end_date = datetime.datetime(*end)
# 銘柄選定基準値
per_reference_value = 10
roe_reference_value = 0.1
# グラフのタイトル
graph_title = "Target: " + "TOPIX500" + "\n"
graph_title += "Reference value: PER " + str(per_reference_value) + "times or less" + "\n"
graph_title += "                        : ROE over " + str(roe_reference_value) + "times" + "\n"
graph_title += "Coverage period: "
graph_title += str(begin[0]) + "/" + str(begin[1]) + "/" + str(begin[2]) + " ~ "
graph_title += str(end[0]) + "/" + str(end[1]) + "/" + str(end[2])


# データの取得
# 既に取得しているデータ部分はコメントアウト済み
# for ticker in tickers:
#     mylib.stock_prices_to_csv(ticker)
# mylib.stock_prices_to_csv(reference_ticker)
# for ticker in tickers:
#     mylib.pl_to_csv(ticker)
# for ticker in tickers:
#     mylib.balance_sheet_to_csv(ticker)
# for ticker in tickers:
#     mylib.sammary_to_csv(ticker)


# 終値データフレーム
closes = mylib.get_stock_prices_dataframe(tickers + [reference_ticker], "Close")
# データ範囲の指定
closes = closes[closes.index >= begin_date]
closes = closes[closes.index <= end_date]


# 当期純利益データフレーム
earnings = mylib.get_earnings_dataframe(tickers + [reference_ticker])
# データ範囲の指定
earnings = earnings[earnings.index <= end_date]


# 自己資本データフレーム
equity = mylib.get_equity_dataframe(tickers + [reference_ticker])
# データ範囲の指定
equity = equity[equity.index <= end_date]


# 発行株数データフレーム
shares = mylib.get_shares(tickers + [reference_ticker])


# EPS(一株当たり利益), ROE(自己資本利益率)のデータフレームの作成
# EPS(一株当たり利益)
eps = earnings / shares.values
# ROE(自己資本利益率)
roe = earnings / equity

# 欠損データの補完
eps = eps.ffill()
roe = roe.ffill()

# 日経平均株価のカラムの削除
eps = eps.drop([reference_ticker], axis = 1)
roe = roe.drop([reference_ticker], axis = 1)


# 終値データフレームの整形, および月次リターンデータフレームの作成
# 月カラムの作成
closes["month"] = closes.index.month
# 月末フラグカラムの作成
closes["end_of_month"] = closes.month.diff().shift(-1)
# 月末のデータのみを抽出
closes = closes[closes.end_of_month != 0]

# 月次リターン(今月の株価と来月の株価の差分)の作成(ラグあり)
monthly_rt = closes.pct_change().shift(-1)
# マーケットリターンの控除
monthly_rt = monthly_rt.sub(monthly_rt[reference_ticker], axis = 0)


# 不要なカラムの削除
closes = closes.drop([reference_ticker, "month", "end_of_month"], axis = 1)
monthly_rt = monthly_rt.drop([reference_ticker, "month", "end_of_month"], axis = 1)


# PER, ROEデータフレームの作成(月次リターンと同次元)
# EPSデータフレーム(月次リターンと同次元)
eps_df = pd.DataFrame(index = monthly_rt.index, columns = monthly_rt.columns)
# ROEデータフレーム(月次リターンと同次元)
roe_df = pd.DataFrame(index = monthly_rt.index, columns = monthly_rt.columns)

# 値の代入
for i in range(len(eps_df)):
    eps_df.iloc[i] = eps[eps.index < eps_df.index[i]].iloc[-1]
for i in range(len(roe_df)):
    roe_df.iloc[i] = roe[roe.index < roe_df.index[i]].iloc[-1]

# PERデータフレーム(月次リターンと同次元)
per_df = closes / eps_df


# データの結合
# 各データを一次元にスタック
stack_monthly_rt = monthly_rt.stack()
stack_per_df = per_df.stack()
stack_roe_df = roe_df.stack()

# データの結合
df = pd.concat([stack_monthly_rt, stack_per_df, stack_roe_df], axis = 1)
# カラム名の設定
df.columns = ["rt", "per", "roe"]

# 異常値の除去
df[df.rt > 1.0] = np.nan


# 割安でクオリティが高い銘柄を抽出
value_df = df[(df.per < per_reference_value) & (df.roe > roe_reference_value)]

# ヒストグラムの描画
plt.figure(figsize=(10.24, 7.68))
plt.hist(value_df["rt"], bins = [(-0.5 + (i / 25.0)) for i in range(26)], ec = "black")
plt.grid(True)
plt.title(graph_title)
plt.xlim(-0.5, 0.5)
plt.xlabel("monthly return")
plt.ylabel("number of trades")
plt.show()


# 累積リターンを作成
balance = value_df.groupby(level = 0).mean().cumsum()

# バランスカーブの描画
plt.close()
plt.figure(figsize=(10.24, 7.68))
plt.plot(balance["rt"])
plt.grid(True)
plt.title(graph_title)
plt.xlabel("date")
plt.ylabel("cumulative return")
plt.show()
