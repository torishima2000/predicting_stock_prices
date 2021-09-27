# 【第六回】機械学習で株予測（3点チャージ法の有効性検証）
# VR（ボリュームレシオ）の計算方法
# サイトにあったプログラムのコピー

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

def vr(df, window=26, type=1):
    """
    Volume Ratio (VR)
    Formula:
    VR[A] = SUM(av + cv/2, n) / SUM(bv + cv/2, n) * 100
    VR[B] = SUM(av + cv/2, n) / SUM(av + bv + cv, n) * 100
    Wako VR = SUM(av - bv - cv, n) / SUM(av + bv + cv, n) * 100
        av = volume if close > pre_close else 0
        bv = volume if close < pre_close else 0
        cv = volume if close = pre_close else 0
    """
    df['av'] = np.where(df['Close'].diff() > 0, df['Volume'], 0)
    avs = df['av'].rolling(window=window, center=False).sum()
    df['bv'] = np.where(df['Close'].diff() < 0, df['Volume'], 0)
    bvs = df['bv'].rolling(window=window, center=False).sum()
    df['cv'] = np.where(df['Close'].diff() == 0, df['Volume'], 0)
    cvs = df['cv'].rolling(window=window, center=False).sum()
    df.drop(['av', 'bv', 'cv'], inplace=True, axis=1)
    if type == 1: # VR[A]
       vr = (avs + cvs / 2) / (bvs + cvs / 2) * 100  
    elif type == 2: # VR[B]
       vr = (avs + cvs / 2) / (avs + bvs + cvs) * 100
    else: # Wako VR
       vr = (avs - bvs - cvs) / (avs + bvs + cvs) * 100
    return vr

def main():
    # 株価データの取得
    df = mylib.get_stock_prices("7203.T")

    # 元データの成形
    begin = datetime.datetime(*[2018, 5, 16])
    end = datetime.datetime(*[2020, 1, 14])
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # VR(Volume Ratio)の算出
    df["VR"] = vr(df, window=10)
    mylib.plot_chart({"VR":df["VR"]})

    print(df)
    


if __name__=="__main__":
    main()
