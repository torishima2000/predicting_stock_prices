# 【第二回】機械学習で株価予測（TA-LibとLightGBMを使った学習モデル構築）
# 学習モデル
# テスト運用

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import talib
from sklearn.model_selection import train_test_split
import mylibrary as mylib

def main():
    # 株価データの取得
    df = mylib.get_stock_prices("7203.T")

    # 元データの成形
    begin = datetime.datetime(*[2005, 7, 1])
    end = datetime.datetime(*[2020, 1, 9])
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # 特徴量の計算
    # 曜日カラムの作成
    df["weekday"] = df.index.weekday

    # 移動平均線の算出
    df["SMA3"] = talib.SMA(np.array(df["Close"]), timeperiod=3)
    df["SMA5"] = talib.SMA(np.array(df["Close"]), timeperiod=5)
    df["SMA25"] = talib.SMA(np.array(df["Close"]), timeperiod=25)
    df["SMA50"] = talib.SMA(np.array(df["Close"]), timeperiod=50)
    df["SMA75"] = talib.SMA(np.array(df["Close"]), timeperiod=75)
    df["SMA100"] = talib.SMA(np.array(df["Close"]), timeperiod=100)

    # ボリンジャーバンドの算出
    df["upper1"], middle, df["lower1"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
    df["upper2"], middle, df["lower2"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)
    df["upper3"], middle, df["lower3"] = talib.BBANDS(np.array(df["Close"]), timeperiod=25, nbdevup=3, nbdevdn=3, matype=0)

    # MACDの算出
    df["MACD"], df["MACDsignal"], df["MACDhist"] = talib.MACD(np.array(df["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)

    # RSIの算出
    df["RSI9"] = talib.RSI(np.array(df["Close"]), timeperiod=9)
    df["RSI14"] = talib.RSI(np.array(df["Close"]), timeperiod=14)


    # 目的変数の作成
    # 3日後の株価の変化量の計算
    df["target"] = df["Open"].diff(-3).shift(-1) * -1

    # 不要インデックスの削除
    df = df.dropna(subset=[ "SMA3", "SMA5", "SMA25", "SMA50", "SMA75", "SMA100",
                            "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
                            "MACD", "MACDsignal", "MACDhist",
                            "RSI9", "RSI14",
                            "target"])

    # 目的変数の型変換
    df["target"] = df["target"].astype(int)

    # 不要カラムの削除
    df = df.drop(["Dividends", "Stock Splits"], axis=1)

    # 欠損値の補完
    df.ffill()

    # 学習データ、テストデータの作成
    # train 学習時データ vaild 学習時の検証データ test 学習後のテストデータ
    df_X = df.drop(["target"], axis=1)
    df_y = np.array(df["target"])
    X_train, X_vaild, y_train, y_vaild = train_test_split(df_X, df_y, train_size=0.8, random_state=0)
    print("X_train:{}".format(X_train.shape))
    print("y_train:{}".format(y_train.shape))
    print("X_vaild:{}".format(X_vaild.shape))
    print("y_vaild:{}".format(y_vaild.shape))

    # 訓練
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("Accuracy:{}".format(score))
    
    # 正解率の確認
    y_pred = model.predict(X_vaild)
    print(y_pred)
    accuracy = sum(y_vaild & y_pred > 0) / len(y_vaild)
    print("Testing Accuracy{}".format(accuracy))

    # テスト運用
    test = df_X
    test = test[test.index >= datetime.datetime(*[2018, 2, 26])]
    test = test[test.index <= datetime.datetime(*[2019, 12, 26])]
    test["isbuy"] = (model.predict(test))
    test["target"] = test["Open"].diff(-3).shift(-1) * -1
    test = test.sort_index()
    test["assets"] = (test["target"]*test["isbuy"]).cumsum()
    mylib.plot_chart({"": test["assets"]})


if __name__=="__main__":
    main()
