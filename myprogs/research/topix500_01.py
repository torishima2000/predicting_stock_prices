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
# データ取得範囲
begin_date = datetime.datetime(2017, 4, 1)
end_date = datetime.datetime(2021, 3, 31)
# 銘柄選定基準値
per_reference_value = 10
roe_reference_value = 0.1


# データの取得
# 既に取得しているデータ部分はコメントアウト済み
# for ticker in tickers:
#     mylib.stock_prices_to_csv(ticker)
# mylib.stock_prices_to_csv("^N225")
# for ticker in tickers:
#     mylib.pl_to_csv(ticker)
# for ticker in tickers:
#     mylib.balance_sheet_to_csv(ticker)
# for ticker in tickers:
#     mylib.sammary_to_csv(ticker)


# 終値データフレームの作成
# 終値
closes = []
# 終値をリストとして記憶
for ticker in tickers:
    df = mylib.get_stock_prices(ticker)
    closes.append(df.Close)
df = mylib.get_stock_prices("^N225")
closes.append(df.Close)

# 終値のリストをDateFrame化
closes = pd.DataFrame(closes).T
# カラム名の指定
closes.columns = [ticker for ticker in tickers] + ["^N225"]
# データのソート
closes = closes.sort_index()
# 欠損データの補完
closes = closes.ffill()
# データ範囲の指定
closes = closes[closes.index >= begin_date]
closes = closes[closes.index <= end_date]


# 当期純利益データフレームの作成
# 当期純利益
earnings = []
# 当期純利益をリストとして記憶
dummy = mylib.get_pl(tickers[0])["Net Income"]
dummy[:] = np.nan
for ticker in tickers:
    df = mylib.get_pl(ticker)
    try:
        earnings.append(df["Net Income"])
    except:
        earnings.append(dummy)
earnings.append(dummy)

# 当期純利益のリストをDateFrame化
earnings = pd.DataFrame(earnings).T
# カラム名の指定
earnings.columns = [ticker for ticker in tickers] + ["^N225"]
# データのソート
earnings = earnings.sort_index()
# データ範囲の指定
earnings = earnings[earnings.index <= end_date]


# 自己資本データフレームの作成
# 自己資本
equity = []
# 自己資本をリストとして記憶
dummy = mylib.get_balance_sheet(tickers[0])["Total Stockholder Equity"]
dummy[:] = np.nan
for ticker in tickers:
    df = mylib.get_balance_sheet(ticker)
    try:
        equity.append(df["Total Stockholder Equity"])
    except:
        equity.append(dummy)
equity.append(dummy)

# 自己資本のリストをDateFrame化
equity = pd.DataFrame(equity).T
# カラム名の指定
equity.columns = [ticker for ticker in tickers] + ["^N225"]
# データのソート
equity = equity.sort_index()
# データ範囲の指定
equity = equity[equity.index <= end_date]


# 発行株数データフレームの作成
# 発行株数
shares = []
# 発行株数をリストとして記憶
for ticker in tickers:
    df = mylib.get_sammary(ticker)
    try:
        shares.append(df["sharesOutstanding"])
    except:
        shares.append(np.nan)
shares.append(np.nan)

# 発行株数のリストをSeries化
shares = pd.Series(shares)
# インデックス名の指定
shares.index = [ticker for ticker in tickers] + ["^N225"]


# EPS(一株当たり利益), ROE(自己資本利益率)のデータフレームの作成
# EPS(一株当たり利益)
eps = earnings / shares.values
# ROE(自己資本利益率)
roe = earnings / equity

# 欠損データの補完
eps = eps.ffill()
roe = roe.ffill()

# 日経平均株価のカラムの削除
eps = eps.drop(["^N225"], axis = 1)
roe = roe.drop(["^N225"], axis = 1)


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
monthly_rt = monthly_rt.sub(monthly_rt["^N225"], axis = 0)


# 不要なカラムの削除
closes = closes.drop(["^N225", "month", "end_of_month"], axis = 1)
monthly_rt = monthly_rt.drop(["^N225", "month", "end_of_month"], axis = 1)


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
plt.hist(value_df["rt"], bins = [-0.4272727, -0.390909, -0.3545454, -0.3181818, -0.2818181, -0.2454545, -0.2090909, -0.1727272, -0.1363636, -0.1, -0.0636363, -0.0272727, 0.0090909, 0.0454545, 0.0818181, 0.1181818, 0.1545454, 0.190909, 0.2272727, 0.2636363, 0.3, 0.3363636, 0.3727272, 0.4090909], ec = "black")
plt.grid(True)
plt.xlim(-0.4, 0.4)
plt.xlabel("monthly return")
plt.ylabel("number of trades")
plt.show()

# 累積リターンを作成
balance = value_df.groupby(level = 0).mean().cumsum()

# バランスカーブの描画
plt.clf()
plt.plot(balance["rt"])
plt.grid(True)
plt.ylim(-0.15, 0.15)
plt.xlabel("date")
plt.ylabel("cumulative return")
plt.show()
