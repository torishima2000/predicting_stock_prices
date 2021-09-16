# 【第二回】機械学習で株価予測（TA-LibとLightGBMを使った学習モデル構築）
# 学習モデル

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
    begin = datetime.datetime(*[2018, 5, 16])
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
    df["variation"] = df["Open"].diff(-3).shift(-1) * -1
    df["target"] = (df["variation"] >= 0)

    # 不要カラムの削除
    df = df.drop(["Dividends", "Stock Splits", "variation"], axis=1)

    # 欠損値の補完
    df.ffill()

    # 訓練データ、教師データの作成
    train_X, vaild_X, train_Y, vaild_Y = train_test_split(df.drop(["target"], axis=1), df["target"], train_size=0.8)

    # データセットを登録
    lgb_train = lgb.Dataset(train_X, train_Y)
    lgb_vaild = lgb.Dataset(vaild_X, vaild_Y)

    # ハイパーパラメータの設定
    lgb_params = {"objective": "binary",}

    # 訓練
    bst = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_vaild])
    lgb.plot_importance(bst, height=0.5, figsize=(10.24, 7.68))
    plt.show()
    plt.close()
    #bst.save_model("model.txt")


if __name__=="__main__":
    main()