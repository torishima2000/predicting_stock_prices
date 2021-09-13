# TOPIX500構成銘柄 S&P500構成銘柄の比較
# 2017/04/01 ~ 2021/03/31
# PER(株価収益率)10倍以下, ROE(自己資本利益率)10%以上

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mylibrary as mylib


# ステータスの設定
# 対象銘柄のリスト
topix500_tickers = mylib.get_codelist_topix500()
topix500_reference_ticker = "^N225"

sp500_tickers = mylib.get_codelist_sp500()
sp500_reference_ticker = "^GSPC"
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
sp500_closes = mylib.get_stock_prices_dataframe(sp500_tickers + [sp500_reference_ticker], "Close")
# データ範囲の指定
sp500_closes = sp500_closes[sp500_closes.index >= begin_date]
sp500_closes = sp500_closes[sp500_closes.index <= end_date]


# TOPIX500の当期純利益データフレーム
topix500_earnings = mylib.get_earnings_dataframe(topix500_tickers + [topix500_reference_ticker])
# データ範囲の指定
topix500_earnings = topix500_earnings[topix500_earnings.index <= end_date]

# S&P500の当期純利益データフレーム
sp500_earnings = mylib.get_earnings_dataframe(sp500_tickers + [sp500_reference_ticker])
# データ範囲の指定
sp500_earnings = sp500_earnings[sp500_earnings.index <= end_date]


# TOPIX500の自己資本データフレーム
topix500_equity = mylib.get_equity_dataframe(topix500_tickers + [topix500_reference_ticker])
# データ範囲の指定
topix500_equity = topix500_equity[topix500_equity.index <= end_date]

# S&P500の自己資本
sp500_equity = mylib.get_equity_dataframe(sp500_tickers + [sp500_reference_ticker])
# データ範囲の指定
sp500_equity = sp500_equity[sp500_equity.index <= end_date]


# TOPIX500の発行株数データフレーム
topix500_shares = mylib.get_shares(topix500_tickers + [topix500_reference_ticker])

# S&P500の発行株数
sp500_shares = mylib.get_shares(sp500_tickers + [sp500_reference_ticker])


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
sp500_eps = sp500_earnings / sp500_shares.values
# ROE(自己資本利益率)
sp500_roe = sp500_earnings / sp500_equity
# 欠損データの補完
sp500_eps = sp500_eps.ffill()
sp500_roe = sp500_roe.ffill()
# 日経平均株価のカラムの削除
sp500_eps = sp500_eps.drop([sp500_reference_ticker], axis = 1)
sp500_roe = sp500_roe.drop([sp500_reference_ticker], axis = 1)


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
sp500_closes["month"] = sp500_closes.index.month
# 月末フラグカラムの作成
sp500_closes["end_of_month"] = sp500_closes.month.diff().shift(-1)
# 月末のデータのみを抽出
sp500_closes = sp500_closes[sp500_closes.end_of_month != 0]
# 月次リターン(今月の株価と来月の株価の差分)の作成(ラグあり)
sp500_monthly_rt = sp500_closes.pct_change().shift(-1)
# マーケットリターンの控除
sp500_monthly_rt = sp500_monthly_rt.sub(sp500_monthly_rt[sp500_reference_ticker], axis = 0)


# 不要なカラムの削除
topix500_closes = topix500_closes.drop([topix500_reference_ticker, "month", "end_of_month"], axis = 1)
topix500_monthly_rt = topix500_monthly_rt.drop([topix500_reference_ticker, "month", "end_of_month"], axis = 1)

# 不要なカラムの削除
sp500_closes = sp500_closes.drop([sp500_reference_ticker, "month", "end_of_month"], axis = 1)
sp500_monthly_rt = sp500_monthly_rt.drop([sp500_reference_ticker, "month", "end_of_month"], axis = 1)


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
sp500_eps_df = pd.DataFrame(index = sp500_monthly_rt.index, columns = sp500_monthly_rt.columns)
# ROEデータフレーム(月次リターンと同次元)
sp500_roe_df = pd.DataFrame(index = sp500_monthly_rt.index, columns = sp500_monthly_rt.columns)
# 値の代入
for i in range(len(sp500_eps_df)):
    sp500_eps_df.iloc[i] = sp500_eps[sp500_eps.index < sp500_eps_df.index[i]].iloc[-1]
for i in range(len(sp500_roe_df)):
    sp500_roe_df.iloc[i] = sp500_roe[sp500_roe.index < sp500_roe_df.index[i]].iloc[-1]
# PERデータフレーム(月次リターンと同次元)
sp500_per_df = sp500_closes / sp500_eps_df


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
sp500_stack_monthly_rt = sp500_monthly_rt.stack()
sp500_stack_per_df = sp500_per_df.stack()
sp500_stack_roe_df = sp500_roe_df.stack()
# データの結合
sp500_df = pd.concat([sp500_stack_monthly_rt, sp500_stack_per_df, sp500_stack_roe_df], axis = 1)
# カラム名の設定
sp500_df.columns = ["rt", "per", "roe"]
# 異常値の除去
sp500_df[sp500_df.rt > 1.0] = np.nan


# 各種指標とその指標に基づいて投資を行った際の結果をまとめるDataFrameオブジェクト
topix500_return_df = pd.DataFrame(columns=["PER", "ROE", "rt"])
sp500_return_df = pd.DataFrame(columns=["PER", "ROE", "rt"])

for i in range(5, 40):
    for j in range(2, 30):
        per_reference_value = i
        roe_reference_value = j / 100.0

        # 割安でクオリティが高い銘柄を抽出
        topix500_value_df = topix500_df[(topix500_df.per < per_reference_value) & (topix500_df.roe > roe_reference_value)]
        # 割安でクオリティが高い銘柄を抽出
        sp500_value_df = sp500_df[(sp500_df.per < per_reference_value) & (sp500_df.roe > roe_reference_value)]

        # 累積リターンを作成
        topix500_balance = topix500_value_df.groupby(level = 0).mean().cumsum()
        sp500_balance = sp500_value_df.groupby(level = 0).mean().cumsum()

        # 結果をDataFrameオブジェクトに追加
        topix500_return_df.loc[str(i) + " " + str(j)] = [per_reference_value, roe_reference_value*100, topix500_balance["rt"].iloc[-2]]
        sp500_return_df.loc[str(i) + " " + str(j)] = [per_reference_value, roe_reference_value*100, sp500_balance["rt"].iloc[-2]]


# 3D散布図を描画
fig = plt.figure(figsize=(10.24, 7.68))
ax = fig.add_subplot(projection="3d")

#ax.tick_params(labelsize=14)
ax.set_xlabel("PER [times]")
#ax.set_xlabel("PER [times]", fontsize=16)
ax.set_ylabel("ROE [%]")
#ax.set_ylabel("ROE [%]", fontsize=16)
ax.set_zlabel("cumulative return")
#ax.set_zlabel("cumulative return", fontsize=16)

ax.scatter(topix500_return_df["PER"], topix500_return_df["ROE"], topix500_return_df["rt"], s=40, marker="o", label="TOPIX500")
ax.scatter(sp500_return_df["PER"], sp500_return_df["ROE"], sp500_return_df["rt"], s=40, marker="o", label="S&P500")
ax.legend()
#ax.legend(fontsize=12)

plt.show()
plt.close()
