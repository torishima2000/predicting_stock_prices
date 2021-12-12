# 二値分類モデル

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import mylibrary as mylib

def main():
    # 変数の定義
    # K-分割交差検証法(k-fold cross-validation)の分割数
    kfold_splits = 11

    # seed値
    seed = 42

    # 証券コード
    security_codes = [
        #2021/12/4
        # 時価総額上位10株
        "7203.T", "6758.T", "6861.T", "6098.T", "9432.T",
        "9984.T", "8035.T", "8306.T", "4063.T", "6594.T",
        # 時価総額上位20株
        "9433.T", "6367.T", "9983.T", "6902.T",
        "7741.T", "6501.T", "7974.T", "4661.T", "4519.T",
        # 日経平均
        "^N225"
    ]
    #security_codes = []

    # データ期間
    begin = datetime.datetime(*[2000, 1, 1])
    end = datetime.datetime(*[2020, 12, 31])
    # テストデータの開始日
    test_begin = datetime.datetime(*[2016, 12, 31])

    # 特徴量
    feature = [
        "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100",
        "EMA3", "EMA5", "EMA15", "EMA25", "EMA50", "EMA75", "EMA100",
        "WMA3", "WMA5", "WMA15", "WMA25", "WMA50", "WMA75", "WMA100",
        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
        "MACD", "MACDsignal", "MACDhist", "MACDGoldenCross",
        "RSI9", "RSI14",
        "VR", "MAER15",
        "ADX", "CCI", "ROC", "ADOSC", "ATR",
        "DoD1", "DoD2", "DoD3",
    ]
    # 削除する特徴量
    drop_feature = [
    ]
    # 特徴量カラムの修正
    for v in drop_feature:
        feature.remove(v)
    # 追加の特徴量
    add_feature = ["N225_" + column for column in feature]
    add_feature += ["DJI_" + column for column in feature]
    add_feature += ["GSPC_" + column for column in feature]

    # 買い判断をするための閾値
    isbuy_threshold = 0.5


    # 株価指標データフレームの作成

    # ダウ平均株価
    # 除外する特徴量
    exclude_feature = drop_feature
    # データの取得
    # mylib.stock_prices_to_csv("^DJI")
    # データをロード
    df_DJI = mylib.get_stock_prices("^DJI")
    # 特徴量の計算
    df_DJI = mylib.colculate_feature(df_DJI, objective=False, exclude=exclude_feature)
    # 整形
    df_DJI = mylib.shaping_yfinance(df_DJI, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_DJI.dropna(subset=(set(feature) - set(exclude_feature)), inplace=True)
    # カラム名の変更
    [df_DJI.rename(columns={columns: "DJI_" + columns}, inplace=True) for columns in df_DJI.columns]

    # ドル円
    # 除外する特徴量
    exclude_feature = ["VR", "ADOSC"] + drop_feature
    # データの取得
    # mylib.stock_prices_to_csv("USDJPY=X")
    # データをロード
    df_USDJPY = mylib.get_stock_prices("USDJPY=X")
    # 特徴量の計算
    df_USDJPY = mylib.colculate_feature(df_USDJPY, objective=False, exclude=exclude_feature)
    # 整形
    df_USDJPY = mylib.shaping_yfinance(df_USDJPY, begin=begin, end=end, drop_columns=["Volume", "Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_USDJPY.dropna(subset=(set(feature) - set(exclude_feature)), inplace=True)
    # カラム名の変更
    [df_USDJPY.rename(columns={columns: "USDJPY_" + columns}, inplace=True) for columns in df_USDJPY.columns]


    # 結果を所持するDataFrame
    result = {
        "params": {},
        "feature importance": {},
        "accuracy": {},             # 正解率 = (TP + TN) / (TP + FP + FN + TP)
        "precision": {},            # 適合率 = TP / (TP + FP)
        "recall": {},               # 再現率 = TP / (TP + FN)
        "f": {},                    # F値 = 2*Precision*Recall / (Precision + Recall)
        "Log loss": {},
        "AUC": {}
    }
    cnt = 0

    # 銘柄群に対して実行
    for security_code in security_codes:

        # 株価データフレームの作成
        # データのダウンロード
        # mylib.stock_prices_to_csv(security_code)
        # 取得したデータの読み取り
        df = mylib.get_stock_prices(security_code)
        # 欠損値がある行の削除
        df.dropna(inplace=True)
        # 特徴量の計算
        df = mylib.colculate_feature(df, objective="binary")
        # データの整形
        df = mylib.shaping_yfinance(df, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"] + drop_feature)
        # 株価指標データの結合
        # df = pd.concat([df, df_N225, df_DJI, df_GSPC, df_USDJPY], axis=1)
        df = pd.concat([df, df_DJI, df_USDJPY], axis=1)
        # 欠損値がある行の削除
        df.dropna(subset=(feature + ["target"]), inplace=True)


        # 学習データ、テストデータの作成
        # train 学習時データ test 学習後のテストデータ
        df_X = df.drop(["growth rate", "target"], axis=1)
        df_y = (df["target"].copy().astype(bool))

        # 1点で分割
        X_train = df_X[df_X.index <= test_begin]
        X_test = df_X[df_X.index > test_begin]
        y_train = df_y[df_y.index <= test_begin]
        y_test = df_y[df_y.index > test_begin]
        Xy_test = df[df.index > test_begin]

        
        Xy_test.insert(len(Xy_test.columns), "predict", (1 * Xy_test["MACDGoldenCross"].copy()))
        Xy_test["predict"] = Xy_test["predict"].copy() - (1e-3 * cnt)
        Xy_test.insert(len(Xy_test.columns), "isbuy", (Xy_test["predict"].copy() >= isbuy_threshold))
        cnt += 1
        # 予測結果の保存
        mylib.isbuy_dataset_to_csv(Xy_test, security_code)


if __name__=="__main__":
    main()
