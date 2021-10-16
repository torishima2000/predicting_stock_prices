# 【第十三回】機械学習を使った株価予測（LightGBMをOptunaでパラメータチューニング）
# ベイズ最適化でのパラメータ探索
# optunaの自動ハイパーパラメータ調整を利用したもの

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna.integration.lightgbm as lgbo
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import mylibrary as mylib


def study_params(X, y, seed, objective="regression", metric="rmse"):
    """ハイパーパラメータの自動チューニングを行う関数

    Args:
        X (Pandas.DataFrame): 特徴量
        y (Pandas.DataFrame): 目的変数
        seed (int): seed値
        objective (str, optional): 目的関数. Defaults to "regression".
        metric (str, optional): 誤差関数. Defaults to "rmse".

    Returns:
        [dict]: 最適化されたハイパーパラメータ
    """
    # ハイパーパラメータのチューニング
    lgb_params = {
        "num_iterations": 1000,         # 木の数
        "max_depth": 7,                 # 木の深さ
        "num_leaves": 31,               # 葉の数
        "min_data_in_leaf": 20,         # 葉に割り当てられる最小データ数
        "boosting": "gbdt",             # 勾配ブースティング
        "objective": objective,         # 回帰
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
    begin = datetime.datetime(*[2011, 1, 1])
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
    term = datetime.datetime(*[2019, 12, 31])
    X_train = df_X[df.index <= term]
    X_test = df_X[df.index > term]
    y_train = df_y[df.index <= term]
    y_test = df_y[df.index > term]

    lgb_test = lgb.Dataset(X_test, label=y_test)


    # ハイパーパラメータの設定
    lgb_params = study_params(X_train, y_train, seed)



    # 回帰モデル
    # K-分割交差検証法(k-fold cross-validation)を行うためのモデル作成
    regression_models = []
    rmse = []
    regression_feature_importance = pd.Series([0] * len(df_X.columns), index=df_X.columns, name="feature importance")
    
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

        # 訓練
        model = lgb.train(params=lgb_params, train_set=lgb_train, valid_sets=[lgb_train, lgb_vaild], verbose_eval=200)

        # モデルの保存
        regression_models.append(model)

        # 訓練結果の評価
        y_pred = model.predict(X_test)
        rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        # 特徴量の重要度の保存
        regression_feature_importance += (model.regression_feature_importance() / kfold_splits)
    
    # 平均スコアの表示
    # スコアの算出方法：平均平方二乗誤差(RMSE)
    ave_rmse = np.average(rmse)
    print(rmse)
    print("average rmse : {}".format(ave_rmse))


    # 特徴量の重みを描画
    # ソート
    regression_feature_importance.sort_values(ascending=True, inplace=True)

    # グラフ描画
    plt.figure(figsize=(10.24, 7.68))
    plt.barh(regression_feature_importance.index, regression_feature_importance)
    #plt.title("")
    #plt.grid(False)
    plt.show()
    plt.close()


    # 株価予測
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
    

    #bst.save_model("model.txt")
    print(lgb_params)


if __name__=="__main__":
    main()
