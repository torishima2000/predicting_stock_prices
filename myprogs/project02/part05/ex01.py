# 【第五回】機械学習で株価予測（Protraを使ったバックテスト解決編）
# ソースコード

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
    # 使用するデータ期間の指定
    begin = datetime.datetime(*[2000, 1, 4])
    end = datetime.datetime(*[2020, 12, 30])


    # 株価データの取得
    # mylib.stock_prices_to_csv("7203.T")
    df = mylib.get_stock_prices("7203.T")


    # 元データの成形
    # データの範囲の調整
    df = df[df.index >= begin]
    df = df[df.index <= end]
    # 不要カラムの削除
    df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)
    # 出来高が欠損している部分の削除
    df = df[df["Volume"] != 0]


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

    # 欠損値のある行の削除
    df.dropna(how="any", axis=0, inplace=True)


    # 目的変数の作成
    # 3日後の株価の変化量の計算
    df["target"] = df["Open"].diff(-3).shift(-1) * -1

    # 欠損値のある行の削除
    df.dropna(subset=["target"], axis=0, inplace=True)
    
    # 目的変数をint型に変換
    df["target"] = df["target"].astype(int)


    # 学習データ、テストデータの作成
    # train 学習時データ vaild 学習時の検証データ test 学習後のテストデータ
    df_X = df.drop(["target"], axis=1)
    df_y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, train_size=0.8, random_state=0)
    X_train, X_vaild, y_train, y_vaild = train_test_split(X_train, y_train, train_size=0.8, random_state=0)


    # データセットを登録
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild)

    # ハイパーパラメータの設定
    lgb_params = {
        "num_iterations": 1000,     # 木の数
        "max_depth": 5,             # 木の深さ
        "num_leaves": 15,           # 葉の数
        "min_data_in_leaf": 30,     # 葉に割り当てられる最小データ数
        "boosting": "gbdt",         # 勾配ブースティング
        "objective": "regression",  # 回帰
        "metric": "rmse",           # 二乗平均平方根誤差
        "learning_rate": 0.1,       # 学習率
        "early_stopping_rounds": 30,# アーリーストッピング
        "force_col_wise": True      # 列毎のヒストグラムの作成を強制する
    }

    # 訓練
    model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], verbose_eval=10)


    # 株価予測
    # テストデータに対するバックテスト
    X_test = X_test.sort_index()
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    X_test["variation"] = y_pred
    X_test = X_test.assign(isbuy=(y_pred >= 10))
    
    # Protra変換部分
    trading_days = {"7203": X_test[X_test["isbuy"] == True]}
    mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))


    # 特徴量の重みの表示
    lgb.plot_importance(model, height=0.5, figsize=(10.24, 7.68))
    #plt.title("")
    #plt.grid(False)
    plt.show()
    plt.close()
    
    #bst.save_model("model.txt")


if __name__=="__main__":
    main()
