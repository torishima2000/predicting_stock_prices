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
            "objective": trial.suggest_categorical("objective", ["binary"]),                    # 二値分類
            "metric": trial.suggest_categorical("metric", ["binary_logloss"]),                  # Log損失
            "boosting": trial.suggest_categorical("boosting", ["gbdt", "dart"]),                # 勾配ブースティング
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),                # 正則化項1
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),                # 正則化項2
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),              # 特徴量の使用割合
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "num_iterations": 1000,                                                             # 木の数
            "max_depth": trial.suggest_int("max_depth", 3, 8),                                  # 木の深さ
            "num_leaves": trial.suggest_int("num_leaves", 3, 255),                              # 葉の数
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),                  # 葉に割り当てられる最小データ数
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),        # 学習率
            "force_col_wise": trial.suggest_categorical("force_col_wise", [True]),              # 列毎のヒストグラムの作成を強制する
            "deterministic": trial.suggest_categorical("force_col_wise", [True])                # 再現性の確保
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
        score = metrics.accuracy_score(y_true=self.y_test, y_pred=(y_pred > 0.6))

        return score



def main():
    # 変数の定義
    # K-分割交差検証法(k-fold cross-validation)の分割数
    kfold_splits = 5

    # seed値
    seed = 42

    # 証券コード
    # security_code = "7203.T"
    security_codes = ["6758.T", "7203.T", "9984.T", "^N225"]

    # データ期間
    begin = datetime.datetime(*[2000, 1, 1])
    end = datetime.datetime(*[2020, 12, 31])
    # テストデータの開始日
    test_begin = datetime.datetime(*[2016, 12, 31])

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
    isbuy_threshold = 0.6


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
    result = {
        "params": {},
        "feature importance": {},
        "assets": pd.DataFrame(),
        "acc": {},
        "Log loss": {},
        "AUC": {}
    }

    # 銘柄群に対して実行
    for security_code in security_codes:

        # 株価データフレームの作成
        # データのダウンロード
        # mylib.stock_prices_to_csv(security_code)
        # 取得したデータの読み取り
        df = mylib.get_stock_prices(security_code)
        # 特徴量の計算
        df = mylib.colculate_feature(df, objective="binary")
        # データの整形
        df = mylib.shaping_yfinance(df, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"] + drop_feature)
        # 欠損値がある行の削除
        df.dropna(subset=(feature + ["target"]), inplace=True)


        # 学習データ、テストデータの作成
        # train 学習時データ test 学習後のテストデータ
        df_X = df.drop(["growth rate", "target"], axis=1)
        df_y = df["target"]

        # 1点で分割
        X_train = df_X[df_X.index <= test_begin]
        X_test = df_X[df_X.index > test_begin]
        y_train = df_y[df_y.index <= test_begin]
        y_test = df_y[df_y.index > test_begin]
        Xy_test = df[df.index > test_begin]


        # optunaによるハイパーパラメータの最適化
        opt = optuna.create_study()
        opt.optimize(Objective(df_X, df_y, seed=seed), n_trials=31)
        result["params"][security_code] = opt.best_params
        result["params"][security_code].update({"num_iterations": 1000})


        # K-分割交差検証法(k-fold cross-validation)の経過保存を行うためのモデル作成
        models = []
        y_preds = []
        log_loss = []
        accuracy_score = []
        auc = []
        result["feature importance"][security_code] = pd.Series([0.0] * len(df_X.columns), index=df_X.columns, name="feature importance of binary")


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
            model = lgb.train(params=result["params"][security_code], train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=-1)

            # モデルの保存
            models.append(model)

            # 特徴量の重要度の保存
            result["feature importance"][security_code] += (model.feature_importance() / kfold_splits)

            # テストデータの予測
            y_preds.append(model.predict(X_test))
            
            # 訓練結果の評価
            # 正解率(Accuracy score)
            accuracy_score.append(metrics.accuracy_score(y_true=y_test, y_pred=(y_preds[-1] > isbuy_threshold)))
            # Log損失(Logarithmic Loss)
            log_loss.append(metrics.log_loss(y_true=y_test, y_pred=y_preds[-1]))
            # AUC(Area Under the Curve)
            auc.append(metrics.roc_auc_score(y_test, y_preds[-1]))


        # 平均スコアの保存
        # 正解率(Accuracy score)
        result["acc"][security_code] = np.lib.average(accuracy_score)
        # Log損失(Logarithmic Loss)
        result["Log loss"][security_code] = np.lib.average(log_loss)
        # AUC(Area Under the Curve)
        result["AUC"][security_code] = np.lib.average(auc)

        """
        # 特徴量の重みを描画
        # ソート
        result["feature importance"][security_code].sort_values(ascending=True, inplace=True)
        # 出力
        print(result["feature importance"][security_code])
        # グラフ描画
        plt.figure(figsize=(10.24, 7.68))
        plt.barh(result["feature importance"][security_code].index, result["feature importance"][security_code])
        plt.title("binary model's feature important")
        #plt.grid(False)
        plt.show()
        plt.close()
        """


        # 株価予測
        # テストデータに対するバックテスト
        X_test.insert(len(X_test.columns), "variation", np.mean(y_preds, axis=0))
        X_test.insert(len(X_test.columns), "isbuy", (X_test["variation"].copy() >= isbuy_threshold))

        # バックテスト
        X_test["growth rate"] = 0
        X_test["total assets"] = 0
        # 株価の増加量の取得
        for index, value in X_test.iterrows():
            X_test.loc[index, "growth rate"] = df.loc[index, "growth rate"]
        # 総資産の移り変わり計算
        total_assets = 10000
        for index, value in X_test.iterrows():
            if value["isbuy"]:
                total_assets -= 1000
                total_assets += 1000 * (1 + value["growth rate"])
            X_test.loc[index, "total assets"] = total_assets
        # 結果の表示
        # mylib.plot_chart({security_code: X_test["total assets"]})


        # 結果の集計
        result["assets"][security_code] = X_test["total assets"]
        
        """
        # Protra変換部分
        trading_days = {security_code: X_test[X_test["isbuy"] == True]}
        mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))
        """



    # 合計値の計算
    result["assets"] = result["assets"].assign(average=(result["assets"].mean(axis=1)))
    result["assets"].to_csv("assets.csv", sep = ",")
    # 異常値の削除
    result["assets"].dropna(axis = 0, inplace=True)
    # 描画用にデータを整形
    plot = {}
    for column, value in result["assets"].iteritems():
        plot[column] = value
    # 資産の変遷の描画
    mylib.plot_chart(plot)

    print(result)



if __name__=="__main__":
    main()
