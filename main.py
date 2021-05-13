# モジュールのインポート
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自作モジュールのインポート
import mylibrary as mylib

# main

# TOPIX500構成銘柄の証券コードを取得
topix500_codes = mylib.get_codelist_topix500()


# 終値データフレームの作成
# 終値
closes = []
# 終値をリストとして記憶
for s in topix500_codes:
    df = mylib.get_stock_prices(str(s) + ".T")
    closes.append(df.Close)

df = mylib.get_stock_prices("^N225")
closes.append(df.Close)

# 終値のリストをDateFrame化
closes = pd.DataFrame(closes).T
# カラム名の指定
closes.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# インデックスのオブジェクト型をObjectからdatetime64[ns]に変換
closes.index = pd.to_datetime(closes.index)
# データのソート
closes = closes.sort_index()
# 欠損データの補完
closes = closes.ffill()
# データ範囲の指定
closes = closes[closes.index <= "2020-9-30"]

# 当期純利益データフレームの作成
# 当期純利益
earnings = []

# 当期純利益をリストとして記憶
dummy = mylib.get_pl(str(topix500_codes[0]) + ".T")["Net Income"]
dummy[:] = np.nan
for s in topix500_codes:
    df = mylib.get_pl(str(s) + ".T")
    try:
        earnings.append(df["Net Income"])
    except:
        earnings.append(dummy)
earnings.append(dummy)

# 当期純利益のリストをDateFrame化
earnings = pd.DataFrame(earnings).T
# カラム名の指定
earnings.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# インデックスのオブジェクト型をObjectからdatetime64[ns]に変換
earnings.index = pd.to_datetime(earnings.index)
# データのソート
earnings = earnings.sort_index()
# データ範囲の指定
earnings = earnings[earnings.index <= "2020-9-30"]


# 自己資本データフレームの作成
# 自己資本
equity = []

# 自己資本をリストとして記憶
dummy = mylib.get_balance_sheet(str(topix500_codes[0]) + ".T")["Total Stockholder Equity"]
dummy[:] = np.nan
for s in topix500_codes:
    df = mylib.get_balance_sheet(str(s) + ".T")
    try:
        equity.append(df["Total Stockholder Equity"])
    except:
        equity.append(dummy)
equity.append(dummy)

# 自己資本のリストをDateFrame化
equity = pd.DataFrame(equity).T
# カラム名の指定
equity.columns = [str(s) + ".T" for s in topix500_codes] + ["^N225"]
# インデックスのオブジェクト型をObjectからdatetime64[ns]に変換
equity.index = pd.to_datetime(equity.index)
# データのソート
equity = equity.sort_index()
# データ範囲の指定
equity = equity[equity.index <= "2020-9-30"]


# 発行株数データフレームの作成
# 発行株数
shares = []

for s in topix500_codes:
    df = mylib.get_sammary(str(s) + ".T")
    try:
        shares.append(df["sharesOutstanding"])
    except:
        shares.append(np.nan)
shares.append(np.nan)

# 発行株数のリストをSeries化
shares = pd.Series(shares)
# インデックス名の指定
shares.index = [str(s) + ".T" for s in topix500_codes] + ["^N225"]


# EPS(一株当たり利益), ROE(自己資本利益率)のデータフレームの作成
# EPS(一株当たり利益)
eps = earnings / shares.values
# ROE(自己資本利益率)
roe = earnings / equity

# 欠損データ
eps = eps.ffill()
roe = roe.ffill()

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

# 2017年4月以降のデータのみ抽出
closes = closes[closes.index > datetime.datetime(2017, 4, 1)]
monthly_rt = monthly_rt[monthly_rt.index > datetime.datetime(2017, 4, 1)]

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
value_df = df[(df.per < 10) & (df.roe > 0.1)]

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
