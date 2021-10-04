# 【第六回】機械学習で株予測（3点チャージ法の有効性検証）
# ボリュームレシオ追加の学習モデル有効性検証

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

def vr_(close, volume, window=26):
    """[summary]

    Args:
        close (np.array): 終値
        volume (np.array): ボリューム
        window (int, optional): ウィンドウサイズ. Defaults to 25.

    Returns:
        [DataFrame.Series]: 期間内の株価上昇日の出来高合計
        [DataFrame.Series]: 期間内の株価下落日の出来高合計
        [DataFrame.Series]: 期間内の株価変わらずの日の出来高合計
    """
    df = pd.DataFrame()
    df["up"] = np.where(np.append(np.nan, np.diff(close)) > 0, volume, 0)
    df["down"] = np.where(np.append(np.nan, np.diff(close)) < 0, volume, 0)
    df["same"] = np.where(np.append(np.nan, np.diff(close)) == 0, volume, 0)
    u = df["up"].rolling(window=window, center=False).sum()
    d = df["down"].rolling(window=window, center=False).sum()
    s = df["same"].rolling(window=window, center=False).sum()
    return u, d, s

def vr_a(close, volume, window):
    """Volume Ratioを計算する関数

    Args:
        close (np.array): 終値
        volume (np.array): ボリューム
        window (int, optional): ウィンドウサイズ
    """
    u, d, s = vr_(close, volume, window)
    vr = (u + s / 2) / (d + s / 2) * 100
    return np.array(vr)

def vr_b(close, volume, window=26):
    """Volume Ratioを計算する関数

    Args:
        close (np.array): 終値
        volume (np.array): ボリューム
        window (int, optional): ウィンドウサイズ. Defaults to 25.
    """
    u, d, s = vr_(close, volume, window)
    vr = (u + s / 2) / (u + d + s) * 100
    return np.array(vr)

def vr_wako(close, volume, window=26):
    """Volume Ratioを計算する関数

    Args:
        close (np.array): 終値
        volume (np.array): ボリューム
        window (int, optional): ウィンドウサイズ. Defaults to 25.
    """
    u, d, s = vr_(close, volume, window)
    vr = (u - d - s) / (u + d + s) * 100
    return np.array(vr)

def three_charge(df):
    return (25 > df["RSI14"]) & (-18 > df["MAER15"]) & (70 > df["VR"])

def main():
    # 株価データの取得
    df = mylib.get_stock_prices("7203.T")

    # 元データの成形
    begin = datetime.datetime(*[1998, 1, 5])
    end = datetime.datetime(*[2020, 1, 24])
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # 特徴量の計算
    # 高値、安値、終値のnp.array化
    high = np.array(df["High"])
    low = np.array(df["Low"])
    close = np.array(df["Close"])
    volume = np.array(df["Volume"]).astype(np.float64)

    # 移動平均線の算出
    df["SMA3"] = talib.SMA(close, timeperiod=3)
    df["SMA5"] = talib.SMA(close, timeperiod=5)
    df["SMA15"] = talib.SMA(close, timeperiod=15)
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

    # VR(Volume Ratio)の算出
    df["VR"] = vr_a(np.array(df["Close"]), np.array(df["Volume"]), window=25)

    # 移動平均乖離率(Moving Average Estrangement Rate)の算出
    sma15 = talib.SMA(close, timeperiod=15)
    df["MAER15"] = 100 * (close - sma15) / sma15

    # 三点チャージ法による判断
    df["charge3"] = three_charge(df)

    # 目的変数の作成
    # 3日後の株価の変化量の計算
    df["target"] = df["Open"].diff(-3).shift(-1) * -1

    # 不要インデックスの削除
    df = df.dropna(subset=[ "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100",
                            "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
                            "MACD", "MACDsignal", "MACDhist",
                            "RSI9", "RSI14",
                            "VR", "MAER15", "charge3",
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
    df_y = df["target"]
    df_X = df_X[df_X.index >= datetime.datetime(*[2006, 2, 20])]
    df_y = df_y[df_y.index >= datetime.datetime(*[2006, 2, 20])]
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
    
    # Protra変換部分
    # テストデータに対するバックテスト
    X_test = X_test.sort_index()
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    X_test["variation"] = y_pred
    X_test = X_test.assign(isbuy=(y_pred >= 10))
    with open(os.path.join("myprogs", "project02", "LightGBM.pt"), mode="w") as f:
        f.write(write_date("7203", X_test[X_test["isbuy"] == True]))

    # 全テストデータに対するバックテスト
    #df["isbuy"] = (model.predict(df_X, num_iteration=model.best_iteration) >= 10)
    #with open(os.path.join("myprogs", "project02", "LightGBM.pt"), mode="w") as f:
    #    f.write(write_date("7203", df[df["isbuy"] == True]))

    # 特徴量の重みの表示
    lgb.plot_importance(model, height=0.5, figsize=(10.24, 7.68))
    #plt.title("")
    #plt.grid(False)
    plt.show()
    plt.close()
    
    #bst.save_model("model.txt")


if __name__=="__main__":
    main()
