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
        self.X_train, self.X_test = train_test_split(df_X, test_size=0.2, shuffle=False)
        self.y_train, self.y_test = train_test_split(df_y, test_size=0.2, shuffle=False)
        self.seed = seed

    def __call__(self, trial):
        """オブジェクトが呼び出されたときに実行"""
        param = {
            "objective": "regression",                                                      # 回帰
            "metric": "rmse",                                                               # 二乗平均平方根誤差
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
        score = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))

        return score



def main():
    # 変数の定義
    # K-分割交差検証法(k-fold cross-validation)の分割数
    kfold_splits = 5

    # seed値
    seed = 41

    # 証券コード
    # security_code = "7203.T"
    security_codes = ["6758.T", "7203.T", "9984.T", "^N225"]

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
    # drop_feature = ["SMA3", "SMA15", "SMA25", "upper1", "upper2", "lower1", "lower2"]
    # 特徴量カラムの修正
    for v in drop_feature:
        feature.remove(v)

    # 買い判断をするための閾値
    isbuy_threshold = 0.05


    # 株価指標データフレームの作成
    # 日経平均株価
    # データの取得
    # mylib.stock_prices_to_csv("^N225")
    # データをロード
    df_N225 = mylib.get_stock_prices("^N225")
    # 特徴量の計算
    df_N225 = mylib.colculate_feature(df_N225, objective=False)
    # 整形
    df_N225 = mylib.shaping_yfinance(df_N225, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_N225.dropna(subset=(feature), inplace=True)
    # カラム名の変更
    [df_N225.rename(columns={columns: "N225_" + columns}, inplace=True) for columns in df_N225.columns]

    # ダウ平均株価
    # データの取得
    # mylib.stock_prices_to_csv("^DJI")
    # データをロード
    df_DJI = mylib.get_stock_prices("^DJI")
    # 特徴量の計算
    df_DJI = mylib.colculate_feature(df_DJI, objective=False)
    # 整形
    df_DJI = mylib.shaping_yfinance(df_DJI, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_DJI.dropna(subset=(feature), inplace=True)
    # カラム名の変更
    [df_DJI.rename(columns={columns: "DJI_" + columns}, inplace=True) for columns in df_DJI.columns]

    # S&P500
    # データの取得
    # mylib.stock_prices_to_csv("^GSPC")
    # データをロード
    df_GSPC = mylib.get_stock_prices("^GSPC")
    # 特徴量の計算
    df_GSPC = mylib.colculate_feature(df_GSPC, objective=False)
    # 整形
    df_GSPC = mylib.shaping_yfinance(df_GSPC, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_GSPC.dropna(subset=(feature), inplace=True)
    # カラム名の変更
    [df_GSPC.rename(columns={columns: "GSPC_" + columns}, inplace=True) for columns in df_GSPC.columns]


    # 結果を所持するDataFrame
    assets = pd.DataFrame()

    # 銘柄群に対して実行
    for security_code in security_codes:

        # 株価データフレームの作成
        # データのダウンロード
        # mylib.stock_prices_to_csv(security_code)
        # 取得したデータの読み取り
        df = mylib.get_stock_prices(security_code)
        # 特徴量の計算
        df = mylib.colculate_feature(df, objective="regression")
        # データの整形
        df = mylib.shaping_yfinance(df, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"] + drop_feature)
        # 欠損値がある行の削除
        df.dropna(subset=(feature + ["target"]), inplace=True)


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


        # ハイパーパラメータの取得
        objective = Objective(df_X, df_y, seed=seed)
        opt = optuna.create_study()
        opt.optimize(objective, n_trials=31)

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
        rmse = []
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
            model = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=-1)

            # モデルの保存
            models.append(model)

            # 特徴量の重要度の保存
            feature_importance += (model.feature_importance() / kfold_splits)

            # 訓練結果の評価
            y_pred = model.predict(X_test)
            rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            #fpr, tpr = roc_curve(y_test, y_pred)
            #auc.append(auc(fpr, tpr))
            
        
        # 平均スコアの表示
        print(rmse)
        ave_rmse = np.average(rmse)
        print("average rmse : {}".format(ave_rmse))
        #ave_auc = np.average(auc)
        #print(auc)
        #print("average auc : {}".format(ave_auc))


        """
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
        """


        # 株価予測
        # テストデータに対するバックテスト
        X_test = X_test.sort_index()
        # モデルを使用し、株価を予測
        X_test["variation"] = 0
        for model_ in models:
            y_pred = model_.predict(X_test.drop("variation", axis=1))
            X_test["variation"] += y_pred / len(models)
        X_test = X_test.assign(isbuy=(y_pred >= isbuy_threshold))

        # バックテスト
        X_test["target"] = 0
        X_test["total assets"] = 0
        # 株価の増加量の記載
        for index, value in X_test.iterrows():
            X_test.loc[index, "target"] = df.loc[index, "target"]
        # 総資産の移り変わりの記憶
        total_assets = 10000
        for index, value in X_test.iterrows():
            if value["isbuy"]:
                total_assets -= 1000
                total_assets += 1000 * (1 + value["target"])
            X_test.loc[index, "total assets"] = total_assets
        # 結果の表示
        # mylib.plot_chart({security_code: X_test["total assets"]})


        # 結果の集計
        assets[security_code] = X_test["total assets"]

        
        """
        # Protra変換部分
        trading_days = {security_code: X_test[X_test["isbuy"] == True]}
        mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))
        """



    # 合計値の計算
    assets["total"] = assets.sum(axis=1)
    assets["total"] /= len(security_codes)
    assets.to_csv("assets.csv", sep = ",")
    # 異常値の削除
    assets.dropna(axis = 0, inplace=True)
    # 描画用にデータを整形
    plot = {}
    for column, value in assets.iteritems():
        plot[column] = value
    # 資産の変遷の描画
    mylib.plot_chart(plot)


if __name__=="__main__":
    main()
