# 【第十三回】機械学習を使った株価予測（LightGBMをOptunaでパラメータチューニング）
# ベイズ最適化でのパラメータ探索
# optunaの自動ハイパーパラメータ調整を利用したもの

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna.integration.lightgbm as lgbo
import talib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import mylibrary as mylib


def main():
    # 変数の定義
    # 層化K-分割交差検証法(stratified k-fold cross-validation)の分割数
    kfold_splits = 10
    # seed値
    seed = 42
    # 証券コード
    security_code = "7203"


    # 株価データフレームの作成
    # データのダウンロード
    # mylib.get_stock_prices(security_code + ".T")
    # 取得したデータの読み取り
    df = mylib.get_stock_prices(security_code + ".T")

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
    df["VR"] = mylib.vr_a(close, volume, window=25)

    # 移動平均乖離率(Moving Average Estrangement Rate)の算出
    sma15 = talib.SMA(close, timeperiod=15)
    df["MAER15"] = 100 * (close - sma15) / sma15

    # 目的変数の作成
    # 3日後の株価の変化量の計算
    df["target"] = df["Open"].diff(-3).shift(-1) * -1

    # 不要インデックスの削除
    df = df.dropna(subset=[ "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100",
                            "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
                            "MACD", "MACDsignal", "MACDhist",
                            "RSI9", "RSI14",
                            "VR", "MAER15",
                            "target"])
    
    # 目的変数の型変換
    df["target"] = df["target"].astype(int)


    # 不要カラムの削除
    df = df.drop(["Dividends", "Stock Splits"], axis=1)

    # 欠損値の補完
    df.ffill()


    # 学習データ、テストデータの作成
    # train 学習時データ test 学習後のテストデータ
    df_X = df.drop(["target"], axis=1)
    df_y = df["target"]
    df_X = df_X[df_X.index >= datetime.datetime(*[2006, 2, 20])]
    df_y = df_y[df_y.index >= datetime.datetime(*[2006, 2, 20])]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, train_size=0.8, random_state=seed)
    lgb_test = lgb.Dataset(X_test, label=y_test)


    # ハイパーパラメータのチューニング
    # ハイパーパラメータの設定
    lgb_params = {
        "num_iterations": 1000,         # 木の数
        "max_depth": 5,                 # 木の深さ
        "num_leaves": 15,               # 葉の数
        "min_data_in_leaf": 30,         # 葉に割り当てられる最小データ数
        "boosting": "gbdt",             # 勾配ブースティング
        "objective": "regression",      # 回帰
        "metric": "rmse",               # 二乗平均平方根誤差
        "learning_rate": 0.1,           # 学習率
        "early_stopping_rounds": 100,   # アーリーストッピング
        "force_col_wise": True,         # 列毎のヒストグラムの作成を強制する
        "deterministic": True,          # 再現性確保用のパラメータ
    }

    # ハイパーパラメータチューニング用のデータセット分割
    X_train_, X_vaild, y_train_, y_vaild = train_test_split(X_train, y_train, train_size=0.75, random_state=seed)
    
    # データセットを登録
    lgb_train = lgb.Dataset(X_train_, label=y_train_)
    lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild)

    # 学習
    study = lgbo.LightGBMTuner(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], optuna_seed=seed)
    study.run()

    # 最適なハイパーパラメータの保存
    lgb_params = study.best_params


    # 層化K-分割交差検証法(stratified k-fold cross-validation)を行うためのモデル作成
    models = []
    scores = []
    feature_importance = pd.Series([0] * len(df_X.columns), index=df_X.columns, name="feature importance")
    
    # KFoldクラスのインスタンス作成
    K_fold = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=seed)

    # 層化K-分割交差検証法(stratified k-fold cross-validation)を用いた学習
    for fold, (train_indices, vaild_indices) in enumerate(K_fold.split(X_train, y_train)):
        # データセットを分割し割り当て
        X_train_ = X_train.iloc[train_indices]
        y_train_ = y_train.iloc[train_indices]
        X_vaild = X_train.iloc[vaild_indices]
        y_vaild = y_train.iloc[vaild_indices]

        # データセットを登録
        lgb_train = lgb.Dataset(X_train_, label=y_train_)
        lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild)

        # 訓練
        model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], verbose_eval=100)

        # モデルの保存
        models.append(model)

        # 訓練結果の評価
        y_pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

        # 特徴量の重要度の保存
        feature_importance += (model.feature_importance() / kfold_splits)
    
    # 平均スコアの表示
    # スコアの算出方法：平均平方二乗誤差(RMSE)
    ave_score = np.average(scores)
    print("average score : {}".format(ave_score))


    # 特徴量の重みを描画
    # ソート
    feature_importance.sort_values(ascending=True, inplace=True)

    # グラフ描画
    plt.figure(figsize=(10.24, 7.68))
    plt.barh(feature_importance.index, feature_importance)
    #plt.title("")
    #plt.grid(False)
    plt.show()
    plt.close()


    # 株価予測
    # テストデータに対するバックテスト
    X_test = X_test.sort_index()
    # モデルを使用し、株価を予測
    X_test["variation"] = 0
    for model_ in models:
        y_pred = model_.predict(X_test.drop("variation", axis=1))
        X_test["variation"] += y_pred / len(models)

    X_test = X_test.assign(isbuy=(y_pred >= 10))
    
    # Protra変換部分
    trading_days = {security_code: X_test[X_test["isbuy"] == True]}
    mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))
    

    #bst.save_model("model.txt")
    print(lgb_params)


if __name__=="__main__":
    main()
