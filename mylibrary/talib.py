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

def colculate_feature(df):
    """特徴量の計算

    Args:
        df (pandas.DataFrame): 始値, 高値, 安値, 終値, 出来高を要素に持つDataFrame

    Returns:
        [pandas.DataFrame]: 特徴量を算出した Pandas.DataFrame
    """
    # 特徴量の計算
    # 高値、安値、終値のnp.array化
    # high = np.array(df["High"])
    # low = np.array(df["Low"])
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

    return df
