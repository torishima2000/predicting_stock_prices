# TOPIX500構成銘柄 S&P500構成銘柄の比較
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
topix500_tickers = mylib.get_codelist_topix500()
topix500_reference_ticker = "^N225"

sandp500_tickers = mylib.get_codelist_sandp500()
sandp500_reference_ticker = "^GSPC"
# データ取得範囲
begin = [2017, 4, 1]
begin_date = datetime.datetime(*begin)
end = [2021, 3, 31]
end_date = datetime.datetime(*end)
# 銘柄選定基準値
per_reference_value = None
roe_reference_value = None
# グラフのタイトル
graph_title = "Target: " + "TOPIX500 and S&P500" + "\n"
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

# for ticker in tickers:
#     mylib.stock_prices_to_csv(ticker)
# mylib.stock_prices_to_csv(reference_ticker)
# for ticker in tickers:
#     mylib.pl_to_csv(ticker)
# for ticker in tickers:
#     mylib.balance_sheet_to_csv(ticker)
# for ticker in tickers:
#     mylib.sammary_to_csv(ticker)


# TOPIX500の終値データフレーム
topix500_closes = mylib.get_stock_prices_dataframe(topix500_tickers + [topix500_reference_ticker], "Close")
# データ範囲の指定
topix500_closes = topix500_closes[topix500_closes.index >= begin_date]
topix500_closes = topix500_closes[topix500_closes.index <= end_date]

# S&P500の終値データフレーム
sandp500_closes = mylib.get_stock_prices_dataframe(sandp500_tickers + [sandp500_reference_ticker], "Close")
# データ範囲の指定
sandp500_closes = sandp500_closes[sandp500_closes.index >= begin_date]
sandp500_closes = sandp500_closes[sandp500_closes.index <= end_date]


# 当期純利益データフレームの作成
# 当期純利益
topix500_earnings = []
# 当期純利益をリストとして記憶
dummy = mylib.get_pl(topix500_tickers[0])["Net Income"]
dummy[:] = np.nan
for ticker in topix500_tickers:
    df = mylib.get_pl(ticker)
    try:
        topix500_earnings.append(df["Net Income"])
    except:
        topix500_earnings.append(dummy)
topix500_earnings.append(dummy)
# 当期純利益のリストをDateFrame化
topix500_earnings = pd.DataFrame(topix500_earnings).T
# カラム名の指定
topix500_earnings.columns = [ticker for ticker in topix500_tickers] + [topix500_reference_ticker]
# データのソート
topix500_earnings = topix500_earnings.sort_index()
# データ範囲の指定
topix500_earnings = topix500_earnings[topix500_earnings.index <= end_date]

# 当期純利益
sandp500_earnings = []
# 当期純利益をリストとして記憶
dummy = mylib.get_pl(sandp500_tickers[0])["Net Income"]
dummy[:] = np.nan
for ticker in sandp500_tickers:
    df = mylib.get_pl(ticker)
    try:
        sandp500_earnings.append(df["Net Income"])
    except:
        sandp500_earnings.append(dummy)
sandp500_earnings.append(dummy)
# 当期純利益のリストをDateFrame化
sandp500_earnings = pd.DataFrame(sandp500_earnings).T
# カラム名の指定
sandp500_earnings.columns = [ticker for ticker in sandp500_tickers] + [sandp500_reference_ticker]
# データのソート
sandp500_earnings = sandp500_earnings.sort_index()
# データ範囲の指定
sandp500_earnings = sandp500_earnings[sandp500_earnings.index <= end_date]


# 自己資本データフレームの作成
# 自己資本
topix500_equity = []
# 自己資本をリストとして記憶
topix500_dummy = mylib.get_balance_sheet(topix500_tickers[0])["Total Stockholder Equity"]
topix500_dummy[:] = np.nan
for ticker in topix500_tickers:
    df = mylib.get_balance_sheet(ticker)
    try:
        topix500_equity.append(df["Total Stockholder Equity"])
    except:
        topix500_equity.append(topix500_dummy)
topix500_equity.append(topix500_dummy)
# 自己資本のリストをDateFrame化
topix500_equity = pd.DataFrame(topix500_equity).T
# カラム名の指定
topix500_equity.columns = [ticker for ticker in topix500_tickers] + [topix500_reference_ticker]
# データのソート
topix500_equity = topix500_equity.sort_index()
# データ範囲の指定
topix500_equity = topix500_equity[topix500_equity.index <= end_date]

# 自己資本
sandp500_equity = []
# 自己資本をリストとして記憶
sandp500_dummy = mylib.get_balance_sheet(sandp500_tickers[0])["Total Stockholder Equity"]
sandp500_dummy[:] = np.nan
for ticker in sandp500_tickers:
    df = mylib.get_balance_sheet(ticker)
    try:
        sandp500_equity.append(df["Total Stockholder Equity"])
    except:
        sandp500_equity.append(sandp500_dummy)
sandp500_equity.append(sandp500_dummy)
# 自己資本のリストをDateFrame化
sandp500_equity = pd.DataFrame(sandp500_equity).T
# カラム名の指定
sandp500_equity.columns = [ticker for ticker in sandp500_tickers] + [sandp500_reference_ticker]
# データのソート
sandp500_equity = sandp500_equity.sort_index()
# データ範囲の指定
sandp500_equity = sandp500_equity[sandp500_equity.index <= end_date]


# 発行株数データフレームの作成
# 発行株数
topix500_shares = []
# 発行株数をリストとして記憶
for ticker in topix500_tickers:
    df = mylib.get_sammary(ticker)
    try:
        topix500_shares.append(df["sharesOutstanding"])
    except:
        topix500_shares.append(np.nan)
topix500_shares.append(np.nan)
# 発行株数のリストをSeries化
topix500_shares = pd.Series(topix500_shares)
# インデックス名の指定
topix500_shares.index = [ticker for ticker in topix500_tickers] + [topix500_reference_ticker]

# 発行株数
sandp500_shares = []
# 発行株数をリストとして記憶
for ticker in sandp500_tickers:
    df = mylib.get_sammary(ticker)
    try:
        sandp500_shares.append(df["sharesOutstanding"])
    except:
        sandp500_shares.append(np.nan)
sandp500_shares.append(np.nan)
# 発行株数のリストをSeries化
sandp500_shares = pd.Series(sandp500_shares)
# インデックス名の指定
sandp500_shares.index = [ticker for ticker in sandp500_tickers] + [sandp500_reference_ticker]


# EPS(一株当たり利益), ROE(自己資本利益率)のデータフレームの作成
# EPS(一株当たり利益)
topix500_eps = topix500_earnings / topix500_shares.values
# ROE(自己資本利益率)
topix500_roe = topix500_earnings / topix500_equity
# 欠損データの補完
topix500_eps = topix500_eps.ffill()
topix500_roe = topix500_roe.ffill()
# 日経平均株価のカラムの削除
topix500_eps = topix500_eps.drop([topix500_reference_ticker], axis = 1)
topix500_roe = topix500_roe.drop([topix500_reference_ticker], axis = 1)

# EPS(一株当たり利益), ROE(自己資本利益率)のデータフレームの作成
# EPS(一株当たり利益)
sandp500_eps = sandp500_earnings / sandp500_shares.values
# ROE(自己資本利益率)
sandp500_roe = sandp500_earnings / sandp500_equity
# 欠損データの補完
sandp500_eps = sandp500_eps.ffill()
sandp500_roe = sandp500_roe.ffill()
# 日経平均株価のカラムの削除
sandp500_eps = sandp500_eps.drop([sandp500_reference_ticker], axis = 1)
sandp500_roe = sandp500_roe.drop([sandp500_reference_ticker], axis = 1)


# 終値データフレームの整形, および月次リターンデータフレームの作成
# 月カラムの作成
topix500_closes["month"] = topix500_closes.index.month
# 月末フラグカラムの作成
topix500_closes["end_of_month"] = topix500_closes.month.diff().shift(-1)
# 月末のデータのみを抽出
topix500_closes = topix500_closes[topix500_closes.end_of_month != 0]
# 月次リターン(今月の株価と来月の株価の差分)の作成(ラグあり)
topix500_monthly_rt = topix500_closes.pct_change().shift(-1)
# マーケットリターンの控除
topix500_monthly_rt = topix500_monthly_rt.sub(topix500_monthly_rt[topix500_reference_ticker], axis = 0)

# 終値データフレームの整形, および月次リターンデータフレームの作成
# 月カラムの作成
sandp500_closes["month"] = sandp500_closes.index.month
# 月末フラグカラムの作成
sandp500_closes["end_of_month"] = sandp500_closes.month.diff().shift(-1)
# 月末のデータのみを抽出
sandp500_closes = sandp500_closes[sandp500_closes.end_of_month != 0]
# 月次リターン(今月の株価と来月の株価の差分)の作成(ラグあり)
sandp500_monthly_rt = sandp500_closes.pct_change().shift(-1)
# マーケットリターンの控除
sandp500_monthly_rt = sandp500_monthly_rt.sub(sandp500_monthly_rt[sandp500_reference_ticker], axis = 0)


# 不要なカラムの削除
topix500_closes = topix500_closes.drop([topix500_reference_ticker, "month", "end_of_month"], axis = 1)
topix500_monthly_rt = topix500_monthly_rt.drop([topix500_reference_ticker, "month", "end_of_month"], axis = 1)

# 不要なカラムの削除
sandp500_closes = sandp500_closes.drop([sandp500_reference_ticker, "month", "end_of_month"], axis = 1)
sandp500_monthly_rt = sandp500_monthly_rt.drop([sandp500_reference_ticker, "month", "end_of_month"], axis = 1)


# PER, ROEデータフレームの作成(月次リターンと同次元)
# EPSデータフレーム(月次リターンと同次元)
topix500_eps_df = pd.DataFrame(index = topix500_monthly_rt.index, columns = topix500_monthly_rt.columns)
# ROEデータフレーム(月次リターンと同次元)
topix500_roe_df = pd.DataFrame(index = topix500_monthly_rt.index, columns = topix500_monthly_rt.columns)
# 値の代入
for i in range(len(topix500_eps_df)):
    topix500_eps_df.iloc[i] = topix500_eps[topix500_eps.index < topix500_eps_df.index[i]].iloc[-1]
for i in range(len(topix500_roe_df)):
    topix500_roe_df.iloc[i] = topix500_roe[topix500_roe.index < topix500_roe_df.index[i]].iloc[-1]
# PERデータフレーム(月次リターンと同次元)
topix500_per_df = topix500_closes / topix500_eps_df

# EPSデータフレーム(月次リターンと同次元)
sandp500_eps_df = pd.DataFrame(index = sandp500_monthly_rt.index, columns = sandp500_monthly_rt.columns)
# ROEデータフレーム(月次リターンと同次元)
sandp500_roe_df = pd.DataFrame(index = sandp500_monthly_rt.index, columns = sandp500_monthly_rt.columns)
# 値の代入
for i in range(len(sandp500_eps_df)):
    sandp500_eps_df.iloc[i] = sandp500_eps[sandp500_eps.index < sandp500_eps_df.index[i]].iloc[-1]
for i in range(len(sandp500_roe_df)):
    sandp500_roe_df.iloc[i] = sandp500_roe[sandp500_roe.index < sandp500_roe_df.index[i]].iloc[-1]
# PERデータフレーム(月次リターンと同次元)
sandp500_per_df = sandp500_closes / sandp500_eps_df


# データの結合
# 各データを一次元にスタック
topix500_stack_monthly_rt = topix500_monthly_rt.stack()
topix500_stack_per_df = topix500_per_df.stack()
topix500_stack_roe_df = topix500_roe_df.stack()
# データの結合
topix500_df = pd.concat([topix500_stack_monthly_rt, topix500_stack_per_df, topix500_stack_roe_df], axis = 1)
# カラム名の設定
topix500_df.columns = ["rt", "per", "roe"]
# 異常値の除去
topix500_df[topix500_df.rt > 1.0] = np.nan

# 各データを一次元にスタック
sandp500_stack_monthly_rt = sandp500_monthly_rt.stack()
sandp500_stack_per_df = sandp500_per_df.stack()
sandp500_stack_roe_df = sandp500_roe_df.stack()
# データの結合
sandp500_df = pd.concat([sandp500_stack_monthly_rt, sandp500_stack_per_df, sandp500_stack_roe_df], axis = 1)
# カラム名の設定
sandp500_df.columns = ["rt", "per", "roe"]
# 異常値の除去
sandp500_df[sandp500_df.rt > 1.0] = np.nan


# 各種指標とその指標に基づいて投資を行った際の結果をまとめるDataFrameオブジェクト
topix500_return_df = pd.DataFrame(columns=["PER", "ROE", "rt"])
sandp500_return_df = pd.DataFrame(columns=["PER", "ROE", "rt"])

for i in range(5, 40):
    for j in range(2, 30):
        per_reference_value = i
        roe_reference_value = j / 100.0

        # 割安でクオリティが高い銘柄を抽出
        topix500_value_df = topix500_df[(topix500_df.per < per_reference_value) & (topix500_df.roe > roe_reference_value)]
        # 割安でクオリティが高い銘柄を抽出
        sandp500_value_df = sandp500_df[(sandp500_df.per < per_reference_value) & (sandp500_df.roe > roe_reference_value)]

        # 累積リターンを作成
        topix500_balance = topix500_value_df.groupby(level = 0).mean().cumsum()
        sandp500_balance = sandp500_value_df.groupby(level = 0).mean().cumsum()

        # 結果をDataFrameオブジェクトに追加
        topix500_return_df.loc[str(i) + " " + str(j)] = [per_reference_value, roe_reference_value*100, topix500_balance["rt"].iloc[-2]]
        sandp500_return_df.loc[str(i) + " " + str(j)] = [per_reference_value, roe_reference_value*100, sandp500_balance["rt"].iloc[-2]]


# 3D散布図を描画
fig = plt.figure(figsize=(10.24, 7.68))
ax = fig.add_subplot(projection="3d")

ax.set_xlabel("PER [times]")
ax.set_ylabel("ROE [%]")
ax.set_zlabel("cumulative return")

ax.scatter(topix500_return_df["PER"], topix500_return_df["ROE"], topix500_return_df["rt"], s=40, marker="o", label="TOPIX500")
ax.scatter(sandp500_return_df["PER"], sandp500_return_df["ROE"], sandp500_return_df["rt"], s=40, marker="o", label="S&P500")
ax.legend()

plt.show()
plt.close()
