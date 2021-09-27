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

def write_date(code, dates):
    """取引日をptファイルに書き込む関数

    Args:
        dates (Datetimeindex): 購入日のみを抽出したデータセット

    Returns:
        [String]: 購入日をもとにした売買基準をprotra用に記述した文字列
    """
    s = "def IsBUYDATE\n"
    s += "  if ((int)Code == " + code + ")\n"
    s += "     if ( \\\n"
    for index, row in dates.iterrows():
        s += "(Year == " + str(index.year)
        s += " && Month == " + str(index.month)
        s += " && Day == " + str(index.day) + ") || \\\n"
    s += "         (Year == 3000))\n"
    s += "         return 1\n"
    s += "     else\n"
    s += "         return 0\n"
    s += "     end\n"
    s += "  end\n"
    s += "end\n"
    return s


def main():
    # 株価データの取得
    df = mylib.get_stock_prices("7203.T")

    # 元データの成形
    begin = datetime.datetime(*[2015, 1, 1])
    end = datetime.datetime(*[2021, 1, 1])
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # 特徴量の計算
    # 高値、安値、終値のnp.array化
    high = np.array(df["High"])
    low = np.array(df["Low"])
    close = np.array(df["Close"])
    volume = np.array(df["Volume"]).astype(np.float64)

    # 曜日カラムの作成
    df["weekday"] = df.index.weekday

    # 移動平均線の算出
    df["SMA3"] = talib.SMA(close, timeperiod=3)
    df["SMA5"] = talib.SMA(close, timeperiod=5)
    df["SMA25"] = talib.SMA(close, timeperiod=25)
    df["SMA50"] = talib.SMA(close, timeperiod=50)
    df["SMA75"] = talib.SMA(close, timeperiod=75)
    df["SMA100"] = talib.SMA(close, timeperiod=100)

    # ボリンジャーバンドの算出
    df["upper1"], middle, df["lower1"] = talib.BBANDS(close, timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
    df["upper2"], middle, df["lower2"] = talib.BBANDS(close, timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)
    df["upper3"], middle, df["lower3"] = talib.BBANDS(close, timeperiod=25, nbdevup=3, nbdevdn=3, matype=0)

    # MACDの算出
    df["MACD"], df["MACDsignal"], df["MACDhist"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # RSIの算出
    df["RSI9"] = talib.RSI(close, timeperiod=9)
    df["RSI14"] = talib.RSI(close, timeperiod=14)

    # ADX(平均方向性指数)の算出
    df["ADX"] = talib.ADX(high, low, close, timeperiod=14)

    # CCI(商品チャンネル指数(Commodity Channel Index) )の算出
    df["CCI"] = talib.CCI(high, low, close, timeperiod=14)

    # ROC(rate of change)の算出
    df["ROC"] = talib.ROC(close, timeperiod=10)

    # ADOSC(チャイキンオシレーター:A/DのMACD)の算出
    df["ADOSC"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    # ATR(Average True Range)の算出
    df["ATR"] = talib.ATR(high, low, close, timeperiod=14)

    # 移動平均乖離率(Moving Average Estrangement Rate)の算出
    sma5 = talib.SMA(close, timeperiod=5)
    sma15 = talib.SMA(close, timeperiod=15)
    sma25 = talib.SMA(close, timeperiod=25)
    df["MAER5"] = 100 * (close - sma5) / sma5
    df["MAER15"] = 100 * (close - sma15) / sma15
    df["MAER25"] = 100 * (close - sma25) / sma25

    # 前日比の算出
    df["DoD1"] = df["Close"] / df["Close"].shift(1)
    df["DoD2"] = df["Close"].shift(1) / df["Close"].shift(2)
    df["DoD3"] = df["Close"].shift(2) / df["Close"].shift(3)

    # VR(volume Ratio)の算出

    print(df)

    # 目的変数の作成
    # 3日後の株価の変化量の計算
    df["target"] = df["Open"].diff(-3).shift(-1) * -1

    # 不要インデックスの削除
    df.dropna(subset=[  "SMA3", "SMA5", "SMA25", "SMA50", "SMA75", "SMA100",
                        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
                        "MACD", "MACDsignal", "MACDhist",
                        "RSI9", "RSI14",
                        "ADX", "CCI", "ROC", "ADOSC",
                        "ATR",
                        "MAER5", "MAER15", "MAER25",
                        "DoD1", "DoD2", "DoD3"],
                        inplace=True)


    # 不要カラムの削除
    df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)

    # 欠損値の補完
    df.ffill()

    # 学習データ、テストデータの作成
    # train 学習時データ vaild 学習時の検証データ test 学習後のテストデータ
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["target"], axis=1), df["target"], train_size=0.8, random_state=0)
    X_train, X_vaild, y_train, y_vaild = train_test_split(X_train, y_train, train_size=0.8, random_state=0)


    # データセットを登録
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild)

    # ハイパーパラメータの設定
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "regression",  # 回帰
        "metric": "mse",            # 二乗誤差関数
        "num_iterations": 1000,     # 学習回数
        "learning_rate": 0.1        # 学習率
    }

    # 訓練
    model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], num_boost_round=100, early_stopping_rounds=30, verbose_eval=10)
    
    # テストデータの推論
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    accuracy = sum(y_test*y_pred > 0) / len(y_test)
    print("Win rate: ", accuracy)

    # テスト運用
    test = X_test
    test["isbuy"] = (y_pred >= 10)
    test["variation"] = y_test
    test = test.sort_index()
    test["assets"] = (test["variation"]*test["isbuy"]).cumsum()
    

    # Protra変換部分
    mylib.plot_chart({"assets": test["assets"]})
    with open(os.path.join("myprogs", "project02", "LightGBM.pt"), mode="w") as f:
        f.write(write_date("^N225", test[test["isbuy"] == True]))

    # 結果の表示
    lgb.plot_importance(model, height=0.5, figsize=(10.24, 7.68))
    plt.show()
    plt.close()
    #bst.save_model("model.txt")


if __name__=="__main__":
    main()