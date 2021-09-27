# 【第六回】機械学習で株予測（3点チャージ法の有効性検証）
# VR（ボリュームレシオ）の計算方法

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

def main():
    # 株価データの取得
    df = mylib.get_stock_prices("7203.T")

    # 元データの成形
    begin = datetime.datetime(*[2018, 5, 16])
    end = datetime.datetime(*[2020, 1, 14])
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # VR(Volume Ratio)の算出
    df["VR"] = vr_a(np.array(df["Close"]), np.array(df["Volume"]), window=10)
    mylib.plot_chart({"VR":df["VR"]})

    print(df)
    


if __name__=="__main__":
    main()
