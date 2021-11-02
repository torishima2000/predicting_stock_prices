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

def colculate_feature(df, objective=None):
    """特徴量の計算

    Args:
        df (pandas.DataFrame): 始値, 高値, 安値, 終値, 出来高を要素に持つDataFrame
        objective (string): 目的関数の種類. This one must be either False or 'regression' or 'binary'.

    Returns:
        [pandas.DataFrame]: 特徴量を算出した Pandas.DataFrame
    """
    # 目的変数の確認
    if (objective is None):
        raise ValueError(
            "'objective' must be given."
        )
    if (objective not in (False, "regression", "binary")):
        raise ValueError("Invalid variable: 'objective' must be either False or 'regression' or 'binary'.")

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
    df["VR"] = vr_a(close, volume, window=25)

    # 移動平均乖離率(Moving Average Estrangement Rate)の算出
    sma15 = talib.SMA(close, timeperiod=15)
    df["MAER15"] = 100 * (close - sma15) / sma15

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

    # 目的変数の計算
    if (objective == "regression"):
        df["target"] = (df["Open"].pct_change(-3).shift(-1) * -1)
    if (objective == "binary"):
        df["growth rate"] = (df["Open"].pct_change(-3).shift(-1) * -1)
        df["target"] = (df["growth rate"] > 0)

    return df
