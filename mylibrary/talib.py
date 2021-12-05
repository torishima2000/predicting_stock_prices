# テクニカル指標算出用のモジュール群

# モジュールのインポート
import pandas as pd
import numpy as np
import talib

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
    df.insert(len(df.columns), "up", np.where(np.append(np.nan, np.diff(close)) > 0, volume, 0))
    df.insert(len(df.columns), "down", np.where(np.append(np.nan, np.diff(close)) < 0, volume, 0))
    df.insert(len(df.columns), "same", np.where(np.append(np.nan, np.diff(close)) == 0, volume, 0))
    u = df["up"].copy().rolling(window=window, center=False).sum()
    d = df["down"].copy().rolling(window=window, center=False).sum()
    s = df["same"].copy().rolling(window=window, center=False).sum()
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

def colculate_feature(df, objective=None, exclude=[],
    feature=[
        "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100", "SMAGoldenCross",
        "EMA3", "EMA5", "EMA15", "EMA25", "EMA50", "EMA75", "EMA100", "EMAGoldenCross",
        "WMA3", "WMA5", "WMA15", "WMA25", "WMA50", "WMA75", "WMA100",
        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
        "MACD", "MACDsignal", "MACDhist", "MACDGoldenCross",
        "RSI9", "RSI14",
        "VR", "MAER15", "MAER25",
        "ADX", "CCI", "ROC", "ADOSC", "ATR",
        "DoD1", "DoD2", "DoD3",
    ]):
    """特徴量の計算
    feature = [
        "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100",
        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
        "MACD", "MACDsignal", "MACDhist",
        "RSI9", "RSI14",
        "VR", "MAER15",
        "ADX", "CCI", "ROC", "ADOSC", "ATR"
        "DoD1", "DoD2", "DoD3"
    ]

    Args:
        df (pandas.DataFrame): 始値, 高値, 安値, 終値, 出来高を要素に持つDataFrame
        objective (string): 目的関数の種類. This one must be either False or 'regression' or 'binary'.
        exclude (list): 除外する特徴量の名称. The element in this list must be included in feature.

    Returns:
        [pandas.DataFrame]: 特徴量を算出した Pandas.DataFrame
    """
    # 目的変数の確認
    if (objective is None):
        raise ValueError(
            "'objective' must be given."
        )
    if (objective not in (False, "regression", "binary")):
        raise ValueError(
            "Invalid variable: 'objective' must be either False or 'regression' or 'binary'."
        )

    # 除外する特徴量の確認
    for v in exclude:
        if (v not in feature):
            raise ValueError("Invalid variable: The element in 'exclude' must be included in feature.")

    # 特徴量の計算
    # 高値、安値、終値のnp.array化
    high = np.array(df["High"].copy())
    low = np.array(df["Low"].copy())
    close = np.array(df["Close"].copy())
    volume = np.array(df["Volume"].copy()).astype(np.float64)

    # 単純移動平均の算出
    if ("SMA3" not in exclude):
        df.insert(len(df.columns), "SMA3", talib.SMA(close, timeperiod=3))
    if ("SMA5" not in exclude):
        df.insert(len(df.columns), "SMA5", talib.SMA(close, timeperiod=5))
    if ("SMA15" not in exclude):
        df.insert(len(df.columns), "SMA15", talib.SMA(close, timeperiod=15))
    if ("SMA25" not in exclude):
        df.insert(len(df.columns), "SMA25", talib.SMA(close, timeperiod=25))
    if ("SMA50" not in exclude):
        df.insert(len(df.columns), "SMA50", talib.SMA(close, timeperiod=50))
    if ("SMA75" not in exclude):
        df.insert(len(df.columns), "SMA75", talib.SMA(close, timeperiod=75))
    if ("SMA100" not in exclude):
        df.insert(len(df.columns), "SMA100", talib.SMA(close, timeperiod=100))
    if ("SMAGoldenCross" not in exclude):
        if ("SMA25" in exclude):
            df.insert(len(df.columns), "SMA25", talib.SMA(close, timeperiod=25))
        if ("SMA75" in exclude):
            df.insert(len(df.columns), "SMA75", talib.SMA(close, timeperiod=75))
        smahist = df["SMA25"].copy() - df["SMA75"].copy()
        df.insert(len(df.columns), "SMAGoldenCross", (1 * ((smahist.copy() >= 0) & (smahist.shift(1).copy() < 0))))
        if ("SMA25" in exclude):
            df.drop(["SMA25"], axis=1, inplace=True)
        if ("SMA75" in exclude):
            df.drop(["SMA75"], axis=1, inplace=True)
    
    # 指数平滑移動平均の算出
    if ("EMA3" not in exclude):
        df.insert(len(df.columns), "EMA3", talib.EMA(close, timeperiod=3))
    if ("EMA5" not in exclude):
        df.insert(len(df.columns), "EMA5", talib.EMA(close, timeperiod=5))
    if ("EMA15" not in exclude):
        df.insert(len(df.columns), "EMA15", talib.EMA(close, timeperiod=15))
    if ("EMA25" not in exclude):
        df.insert(len(df.columns), "EMA25", talib.EMA(close, timeperiod=25))
    if ("EMA50" not in exclude):
        df.insert(len(df.columns), "EMA50", talib.EMA(close, timeperiod=50))
    if ("EMA75" not in exclude):
        df.insert(len(df.columns), "EMA75", talib.EMA(close, timeperiod=75))
    if ("EMA100" not in exclude):
        df.insert(len(df.columns), "EMA100", talib.EMA(close, timeperiod=100))
    if ("EMAGoldenCross" not in exclude):
        if ("EMA25" in exclude):
            df.insert(len(df.columns), "EMA25", talib.EMA(close, timeperiod=25))
        if ("EMA75" in exclude):
            df.insert(len(df.columns), "EMA75", talib.EMA(close, timeperiod=75))
        emahist = df["EMA25"].copy() - df["EMA75"].copy()
        df.insert(len(df.columns), "EMAGoldenCross", (1 * ((emahist.copy() >= 0) & (emahist.shift(1).copy() < 0))))
        if ("EMA25" in exclude):
            df.drop(["EMA25"], axis=1, inplace=True)
        if ("EMA75" in exclude):
            df.drop(["EMA75"], axis=1, inplace=True)

    # 加重移動平均の算出
    if ("WMA3" not in exclude):
        df.insert(len(df.columns), "WMA3", talib.WMA(close, timeperiod=3))
    if ("WMA5" not in exclude):
        df.insert(len(df.columns), "WMA5", talib.WMA(close, timeperiod=5))
    if ("WMA15" not in exclude):
        df.insert(len(df.columns), "WMA15", talib.WMA(close, timeperiod=15))
    if ("WMA25" not in exclude):
        df.insert(len(df.columns), "WMA25", talib.WMA(close, timeperiod=25))
    if ("WMA50" not in exclude):
        df.insert(len(df.columns), "WMA50", talib.WMA(close, timeperiod=50))
    if ("WMA75" not in exclude):
        df.insert(len(df.columns), "WMA75", talib.WMA(close, timeperiod=75))
    if ("WMA100" not in exclude):
        df.insert(len(df.columns), "WMA100", talib.WMA(close, timeperiod=100))

    # ボリンジャーバンドの算出
    upper1, middle, lower1 = talib.BBANDS(close, timeperiod=25, nbdevup=1, nbdevdn=1, matype=0)
    if ("upper1" not in exclude):
        df.insert(len(df.columns), "upper1", upper1)
    if ("lower1" not in exclude):
        df.insert(len(df.columns), "lower1", lower1)
    upper2, middle, lower2 = talib.BBANDS(close, timeperiod=25, nbdevup=2, nbdevdn=2, matype=0)
    if ("upper2" not in exclude):
        df.insert(len(df.columns), "upper2", upper2)
    if ("lower2" not in exclude):
        df.insert(len(df.columns), "lower2", lower2)
    upper3, middle, lower3 = talib.BBANDS(close, timeperiod=25, nbdevup=3, nbdevdn=3, matype=0)
    if ("upper3" not in exclude):
        df.insert(len(df.columns), "upper3", upper3)
    if ("lower3" not in exclude):
        df.insert(len(df.columns), "lower3", lower3)

    # MACDの算出
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    if ("MACD" not in exclude):
        df.insert(len(df.columns), "MACD", macd)
    if ("MACDsignal" not in exclude):
        df.insert(len(df.columns), "MACDsignal", macdsignal)
    if ("MACDhist" not in exclude):
        df.insert(len(df.columns), "MACDhist", macdhist)
    if ("MACDGoldenCross" not in exclude):
        if ("MACDhist" in exclude):
            df.insert(len(df.columns), "MACDhist", macdhist)
        df.insert(len(df.columns), "MACDGoldenCross", 1 * ((df["MACDhist"].copy() >= 0) & (df["MACDhist"].shift(1).copy() < 0)))
        if ("MACDhist" in exclude):
            df.drop(["MACDhist"], axis=1, inplace=True)

    # RSIの算出
    if ("RSI9" not in exclude):
        df.insert(len(df.columns), "RSI9", talib.RSI(close, timeperiod=9))
    if ("RSI14" not in exclude):
        df.insert(len(df.columns), "RSI14", talib.RSI(close, timeperiod=14))

    # VR(Volume Ratio)の算出
    if ("VR" not in exclude):
        df.insert(len(df.columns), "VR", vr_a(close, volume, window=25))

    # 移動平均乖離率(Moving Average Estrangement Rate)の算出
    if ("MAER15" not in exclude):
        sma15 = talib.SMA(close, timeperiod=15)
        df.insert(len(df.columns), "MAER15", (100 * (close - sma15) / sma15))
    if ("MAER25" not in exclude):
        sma25 = talib.SMA(close, timeperiod=25)
        df.insert(len(df.columns), "MAER25", (100 * (close - sma25) / sma25))

    # ADX(平均方向性指数)の算出
    if ("ADX" not in exclude):
        df.insert(len(df.columns), "ADX", talib.ADX(high, low, close, timeperiod=14))

    # CCI(商品チャンネル指数(Commodity Channel Index) )の算出
    if ("CCI" not in exclude):
        df.insert(len(df.columns), "CCI", talib.CCI(high, low, close, timeperiod=14))

    # ROC(rate of change)の算出
    if ("ROC" not in exclude):
        df.insert(len(df.columns), "ROC", talib.ROC(close, timeperiod=10))

    # ADOSC(チャイキンオシレーター:A/DのMACD)の算出
    if ("ADOSC" not in exclude):
        df.insert(len(df.columns), "ADOSC", talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10))

    # ATR(Average True Range)の算出
    if ("ATR" not in exclude):
        df.insert(len(df.columns), "ATR", talib.ATR(high, low, close, timeperiod=14))

    # 前日比(Day over Day)の算出
    if ("DoD1" not in exclude):
        df.insert(len(df.columns), "DoD1", (df["Open"] / df["Open"].shift(1)))
    if ("DoD2" not in exclude):
        df.insert(len(df.columns), "DoD2", (df["Open"] / df["Open"].shift(2)))
    if ("DoD3" not in exclude):
        df.insert(len(df.columns), "DoD3", (df["Open"] / df["Open"].shift(3)))

    # 目的変数の計算
    if (objective == "regression"):
        # 1日後の始値から4日後の始値までの変化率
        # (4 - 1) / 1
        df.insert(len(df.columns), "target", (df["Open"].pct_change(-3).shift(-1) * -1))
    if (objective == "binary"):
        df.insert(len(df.columns), "growth rate", (df["Open"].pct_change(3).shift(-4)))
        df.insert(len(df.columns), "target", (df["growth rate"].copy() > 0))

    return df
