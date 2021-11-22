# 二値分類モデル

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import mylibrary as mylib


class Objective:
    """目的関数に相当するクラス"""

    def __init__(self, df_X, df_y, seed=42):
        """インスタンス作成時に一度だけ実行"""
        self.X_train, self.X_test = train_test_split(df_X, test_size=0.2, shuffle=False)
        self.y_train, self.y_test = train_test_split(df_y, test_size=0.2, shuffle=False)
        self.seed = seed

    def __call__(self, trial):
        """オブジェクトが呼び出されたときに実行"""
        param = {
            "objective": "binary",                                                          # 回帰
            "metric": "binary_logloss",                                                     # 二乗平均平方根誤差
            "boosting": trial.suggest_categorical("boosting", ["gbdt", "dart"]),            # 勾配ブースティング
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),            # 正則化項1
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),            # 正則化項2
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),          # 特徴量の使用割合
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "num_iterations": 1000,                                                         # 木の数
            "max_depth": trial.suggest_int("max_depth", 3, 8),                              # 木の深さ
            "num_leaves": trial.suggest_int("num_leaves", 3, 255),                          # 葉の数
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),              # 葉に割り当てられる最小データ数
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),    # 学習率
            "early_stopping_rounds": 100,                                                   # アーリーストッピング
            "force_col_wise": True,                                                         # 列毎のヒストグラムの作成を強制する
        }

        # ハイパーパラメータチューニング用のデータセット分割
        X_train, X_valid, y_train, y_valid = train_test_split(self.X_train, self.y_train, train_size=0.75, random_state=self.seed)

        # データセットを登録
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)

        # 学習モデル作成
        model = lgb.train(params=param, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=-1)

        # モデルでの予測
        y_pred = model.predict(self.X_test)
        score = metrics.accuracy_score(y_true=self.y_test, y_pred=(y_pred > 0.5))

        return score



def main():
    # 変数の定義
    # K-分割交差検証法(k-fold cross-validation)の分割数
    kfold_splits = 5

    # seed値
    seed = 42

    # 証券コード
    security_code = "7203.T"

    # データ期間
    begin = datetime.datetime(*[2000, 1, 1])
    end = datetime.datetime(*[2020, 12, 31])

    # 特徴量
    feature = [
        "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100",
        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
        "MACD", "MACDsignal", "MACDhist",
        "RSI9", "RSI14",
        "VR", "MAER15",
        "ADX", "CCI", "ROC", "ADOSC", "ATR"
    ]
    # 削除する特徴量
    drop_feature = []

    # 買い判断をするための閾値
    isbuy_threshold = 0.8


    # 株価データフレームの作成
    # データのダウンロード
    # mylib.get_stock_prices(security_code)
    # 取得したデータの読み取り
    df = mylib.get_stock_prices(security_code)

    # 不要カラムの削除
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

    # 元データの成形
    df = df[df.index >= begin]
    df = df[df.index <= end]


    # 特徴量の計算
    df = mylib.colculate_feature(df, objective="binary")

    # 特徴量の削除
    # dfからの修正
    df.drop(drop_feature, axis = 1, inplace=True)
    # 特徴量カラムの修正
    for v in drop_feature:
        feature.remove(v)
    
    # 目的変数の作成
    # 目的変数の計算
    df["growth rate"] = (df["Open"].pct_change(-3).shift(-1) * -1)
    df["target"] = (df["growth rate"] > 0)

    # 欠損値がある行の削除
    df.dropna(subset=(feature + ["target"]), inplace=True)

    # 欠損値の補完
    df.ffill()

    # 学習データ、テストデータの作成
    # train 学習時データ test 学習後のテストデータ
    df_X = df.drop(["growth rate", "target"], axis=1)
    df_y = df["target"]

    # 1点で分割
    term = datetime.datetime(*[2016, 12, 31])
    X_train = df_X[df.index <= term]
    X_test = df_X[df.index > term]
    y_train = df_y[df.index <= term]
    y_test = df_y[df.index > term]

    lgb_test = lgb.Dataset(X_test, label=y_test)


    # ハイパーパラメータの取得
    objective = Objective(df_X, df_y, seed=seed)
    opt = optuna.create_study()
    opt.optimize(objective, n_trials=15)

    params = {
        "objective": "binary",                                                              # 回帰
        "metric": "binary_logloss",                                                         # 二乗平均平方根誤差
        "num_iterations": 1000,                                                             # 木の数
        "early_stopping_rounds": 100,                                                       # アーリーストッピング
        "force_col_wise": True,                                                             # 列毎のヒストグラムの作成を強制する
        "deterministic": True                                                               # 再現性確保用のパラメータ
    }
    params.update(opt.best_params)


    # K-分割交差検証法(k-fold cross-validation)を行うためのモデル作成
    models = []
    accuracy_score = []
    auc = []
    feature_importance = pd.Series([0.0] * len(df_X.columns), index=df_X.columns, name="feature importance of binary")
    

    # KFoldクラスのインスタンス作成
    K_fold = KFold(n_splits=kfold_splits, shuffle=True, random_state=seed)

    # K-分割交差検証法(k-fold cross-validation)を用いた学習
    for fold, (train_indices, valid_indices) in enumerate(K_fold.split(X_train, y_train)):
        # データセットを分割し割り当て
        X_train_ = X_train.iloc[train_indices]
        y_train_ = y_train.iloc[train_indices]
        X_valid = X_train.iloc[valid_indices]
        y_valid = y_train.iloc[valid_indices]

        # データセットを登録
        lgb_train = lgb.Dataset(X_train_, label=y_train_)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid)

        # 訓練
        model = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=200)

        # モデルの保存
        models.append(model)

        # 特徴量の重要度の保存
        feature_importance += (model.feature_importance() / kfold_splits)

        # 訓練結果の評価
        y_pred = model.predict(X_test)
        accuracy_score.append(metrics.accuracy_score(y_true=y_test, y_pred=(y_pred > 0.5)))
    

     # 平均スコアの表示
    print(accuracy_score)
    ave_accuracy_score = np.average(accuracy_score)
    print("average accuracy score : {}".format(ave_accuracy_score))


    # 特徴量の重みを描画
    # ソート
    feature_importance.sort_values(ascending=True, inplace=True)
    # 出力
    print(feature_importance)
    # グラフ描画
    plt.figure(figsize=(10.24, 7.68))
    plt.barh(feature_importance.index, feature_importance)
    plt.title("binary model's feature important")
    #plt.grid(False)
    plt.show()
    plt.close()



if __name__=="__main__":
    main()
