# 

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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import mylibrary as mylib


def study_params(X, y, seed, objective="regression", metric="rmse"):
    """ハイパーパラメータの自動チューニングを行う関数

    Args:
        X (Pandas.DataFrame): 特徴量
        y (Pandas.DataFrame): 目的変数
        seed (int): seed値

    Returns:
        [dict]: 最適化されたハイパーパラメータ
    """
    # ハイパーパラメータのチューニング
    lgb_params = {
        "num_iterations": 1000,         # 木の数
        "max_depth": 5,                 # 木の深さ
        "num_leaves": 15,               # 葉の数
        "min_data_in_leaf": 20,         # 葉に割り当てられる最小データ数
        "boosting": "gbdt",             # 勾配ブースティング
        "objective": objective,      # 回帰
        "metric": metric,               # 二乗平均平方根誤差
        "learning_rate": 0.01,          # 学習率
        "early_stopping_rounds": 100,   # アーリーストッピング
        "force_col_wise": True,         # 列毎のヒストグラムの作成を強制する
        "deterministic": True,          # 再現性確保用のパラメータ
    }

    # ハイパーパラメータチューニング用のデータセット分割
    X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, train_size=0.75, random_state=seed)
    
    # データセットを登録
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild)

    # 学習
    study = lgbo.LightGBMTuner(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], optuna_seed=seed, verbose_eval=200)
    study.run()

    return study.best_params


def main():
    # 変数の定義
    # K-分割交差検証法(k-fold cross-validation)の分割数
    kfold_splits = 10
    # seed値
    seed = 42
    # 証券コード
    security_code = "7203"
    # データ期間
    begin = datetime.datetime(*[2000, 1, 1])
    end = datetime.datetime(*[2020, 12, 31])
    # 特徴量
    feature = [ "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100",
                "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
                "MACD", "MACDsignal", "MACDhist",
                "RSI9", "RSI14",
                "VR", "MAER15"]
    # 削除する特徴量
    drop_feature = ["SMA3", "SMA15", "SMA25", "upper1", "upper2", "lower1", "lower2"]


    # 株価データフレームの作成
    # データのダウンロード
    # mylib.get_stock_prices(security_code + ".T")
    # 取得したデータの読み取り
    df = mylib.get_stock_prices(security_code + ".T")

    # 不要カラムの削除
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

    # 元データの成形
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # 特徴量の計算
    df = mylib.colculate_feature(df)

    # 不要な特徴量の削除
    df.drop(drop_feature, axis = 1, inplace=True)
    for v in drop_feature:
        feature.remove(v)

    # 目的変数の作成
    # 3日後の株価の変化量の計算
    df["target"] = df["Open"].diff(-3).shift(-1) * -1

    # 欠損値がある行の削除
    df.dropna(subset=(feature + ["target"]), inplace=True)
    
    # 目的変数の型変換
    df["target"] = df["target"].astype(int)

    # 欠損値の補完
    df.ffill()


    # 学習データ、テストデータの作成
    # train 学習時データ test 学習後のテストデータ
    df_X = df.drop(["target"], axis=1)
    df_y = df["target"]

    # 1点で分割
    term = datetime.datetime(*[2016, 12, 31])
    X_train = df_X[df.index <= term]
    X_test = df_X[df.index > term]
    y_train = df_y[df.index <= term]
    y_test = df_y[df.index > term]

    lgb_test = lgb.Dataset(X_test, label=y_test)


    # K-分割交差検証法(k-fold cross-validation)を行うためのモデル作成
    # 回帰モデル
    regression_models = []
    rmse = []
    regression_feature_importance = pd.Series([0] * len(df_X.columns), index=df_X.columns, name="feature importance of regression")
    
    # 二値分類モデル
    binary_models = []
    auc = []
    binary_feature_importance = pd.Series([0] * len(df_X.columns), index=df_X.columns, name="feature importance of binary")
    

    # KFoldクラスのインスタンス作成
    K_fold = KFold(n_splits=kfold_splits, shuffle=True, random_state=seed)

    # K-分割交差検証法(k-fold cross-validation)を用いた学習
    for fold, (train_indices, vaild_indices) in enumerate(K_fold.split(X_train, y_train)):
        # データセットを分割し割り当て
        X_train_ = X_train.iloc[train_indices]
        y_train_ = y_train.iloc[train_indices]
        X_vaild = X_train.iloc[vaild_indices]
        y_vaild = y_train.iloc[vaild_indices]

        # データセットを登録
        lgb_train = lgb.Dataset(X_train_, label=y_train_)
        lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild)


        # 回帰モデル
        # 訓練
        regression_model = lgb.train(params=study_params(X_train_, y_train_, seed), train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], verbose_eval=200)

        # モデルの保存
        regression_models.append(regression_model)

        # 訓練結果の評価
        regression_y_pred = regression_model.predict(X_test)
        rmse.append(np.sqrt(mean_squared_error(y_test, regression_y_pred)))

        # 特徴量の重要度の保存
        regression_feature_importance += (regression_model.feature_importance() / kfold_splits)


        # 二値分類モデル
        # 訓練
        binary_model = lgb.train(params=study_params(X_train_, y_train_, seed, objective="binary", metric="binary_logloss"), train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], verbose_eval=200)

        # モデルの保存
        binary_models.append(binary_model)

        # 訓練結果の評価
        binary_y_pred = binary_model.predict(X_test)
        #fpr, tpr = roc_curve(y_test, binary_y_pred)
        #auc.append(auc(fpr, tpr))
    
    # 平均スコアの表示
    # 回帰モデル
    # スコアの算出方法：平均平方二乗誤差(RMSE)
    ave_rmse = np.average(rmse)
    print(rmse)
    print("average rmse : {}".format(ave_rmse))

    # 二値分類モデル
    #ave_auc = np.average(auc)
    #print(auc)
    #print("average auc : {}".format(ave_auc))


    # 特徴量の重みを描画
    # 回帰モデル
    # ソート
    regression_feature_importance.sort_values(ascending=True, inplace=True)
    # グラフ描画
    plt.figure(figsize=(10.24, 7.68))
    plt.barh(regression_feature_importance.index, regression_feature_importance)
    plt.title("regression model's feature important")
    #plt.grid(False)
    plt.show()
    plt.close()

    # 二値分類モデル
    # ソート
    #binary_feature_importance.sort_values(ascending=True, inplace=True)
    # グラフ描画
    #plt.figure(figsize=(10.24, 7.68))
    #plt.barh(binary_feature_importance.index, binary_feature_importance)
    #plt.title("binary model's feature important")
    #plt.grid(False)
    #plt.show()
    #plt.close()


    # 株価予測
    # 回帰モデル
    # テストデータに対するバックテスト
    X_test = X_test.sort_index()
    # モデルを使用し、株価を予測
    X_test["variation"] = 0
    for model_ in regression_models:
        y_pred = model_.predict(X_test.drop("variation", axis=1))
        X_test["variation"] += y_pred / len(regression_models)
    X_test = X_test.assign(isbuy=(y_pred >= 5))
    # Protra変換部分
    trading_days = {security_code: X_test[X_test["isbuy"] == True]}
    mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))

    # 二値分類モデル
    # モデルを使用し、株価を予測
    X_test["variation"] = 0
    for model_ in binary_models:
        y_pred = model_.predict(X_test.drop("variation", axis=1))
        X_test["variation"] += y_pred / len(binary_models)
    X_test = X_test.assign(isbuy=(y_pred >= 0.7))
    # Protra変換部分
    trading_days = {security_code: X_test[X_test["isbuy"] == True]}
    mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))

    #bst.save_model("model.txt")


if __name__=="__main__":
    main()
