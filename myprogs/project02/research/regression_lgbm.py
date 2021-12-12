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
            "objective": trial.suggest_categorical("objective", ["regression"]),                # 回帰
            "metric": trial.suggest_categorical("metric", ["rmse"]),                            # 二乗平均平方根誤差
            "boosting": trial.suggest_categorical("boosting", ["gbdt"]),                        # 勾配ブースティング
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
        model = lgb.train(params=param, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=False)

        # モデルでの予測
        y_pred = model.predict(self.X_test)
        score = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))

        return score



def main():
    # 変数の定義
    # K-分割交差検証法(k-fold cross-validation)の分割数
    kfold_splits = 11

    # seed値
    seed = 42

    # 証券コード
    security_codes = [
        # 時価総額上位10株
        "4063.T", "6098.T", "6861.T", "6758.T", "7203.T",
        "8035.T", "8306.T", "9432.T", "9433.T", "9984.T",
        # 時価総額上位20株
        "4519.T", "4661.T", "6367.T", "6501.T", "6594.T",
        "6902.T", "7741.T", "7974.T", "9983.T",
        # 日経平均
        "^N225"
    ]
    # security_codes = []

    # データ期間
    begin = datetime.datetime(*[2000, 1, 1])
    end = datetime.datetime(*[2020, 12, 31])
    # テストデータの開始日
    test_begin = datetime.datetime(*[2016, 12, 31])

    # 特徴量
    feature = [
        "SMA3", "SMA5", "SMA15", "SMA25", "SMA50", "SMA75", "SMA100", "SMAGoldenCross",
        "EMA3", "EMA5", "EMA15", "EMA25", "EMA50", "EMA75", "EMA100", "EMAGoldenCross",
        "WMA3", "WMA5", "WMA15", "WMA25", "WMA50", "WMA75", "WMA100",
        "MAER15", "MAER25",
        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
        "MACD", "MACDsignal", "MACDhist", "MACDGoldenCross",
        "RSI9", "RSI14",
        "VR",
        "ADX", "CCI", "ROC", "ADOSC", "ATR",
        "DoD1", "DoD2", "DoD3",
    ]
    # 削除する特徴量
    drop_feature = [
    ]
    # 特徴量カラムの修正
    for v in drop_feature:
        feature.remove(v)

    # 買い判断をするための閾値
    isbuy_threshold = 0


    # 株価指標データフレームの作成
    """
    # 日経平均株価
    # 除外する特徴量
    exclude_feature = drop_feature
    # データの取得
    # mylib.stock_prices_to_csv("^N225")
    # データをロード
    df_N225 = mylib.get_stock_prices("^N225")
    # 特徴量の計算
    df_N225 = mylib.colculate_feature(df_N225, objective=False, exclude=exclude_feature)
    # 整形
    df_N225 = mylib.shaping_yfinance(df_N225, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_N225.dropna(subset=(set(feature) - set(exclude_feature)), inplace=True)
    # カラム名の変更
    [df_N225.rename(columns={columns: "N225_" + columns}, inplace=True) for columns in df_N225.columns]
    """

    # ダウ平均株価
    # 除外する特徴量
    exclude_feature = drop_feature
    # データの取得
    # mylib.stock_prices_to_csv("^DJI")
    # データをロード
    df_DJI = mylib.get_stock_prices("^DJI")
    # 特徴量の計算
    df_DJI = mylib.colculate_feature(df_DJI, objective=False, exclude=exclude_feature)
    # 整形
    df_DJI = mylib.shaping_yfinance(df_DJI, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_DJI.dropna(subset=(set(feature) - set(exclude_feature)), inplace=True)
    # カラム名の変更
    [df_DJI.rename(columns={columns: "DJI_" + columns}, inplace=True) for columns in df_DJI.columns]

    """
    # S&P500
    # 除外する特徴量
    exclude_feature = drop_feature
    # データの取得
    # mylib.stock_prices_to_csv("^GSPC")
    # データをロード
    df_GSPC = mylib.get_stock_prices("^GSPC")
    # 特徴量の計算
    df_GSPC = mylib.colculate_feature(df_GSPC, objective=False, exclude=exclude_feature)
    # 整形
    df_GSPC = mylib.shaping_yfinance(df_GSPC, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_GSPC.dropna(subset=(set(feature) - set(exclude_feature)), inplace=True)
    # カラム名の変更
    [df_GSPC.rename(columns={columns: "GSPC_" + columns}, inplace=True) for columns in df_GSPC.columns]
    """

    # ドル円
    # 除外する特徴量
    exclude_feature = ["VR", "ADOSC"] + drop_feature
    # データの取得
    # mylib.stock_prices_to_csv("USDJPY=X")
    # データをロード
    df_USDJPY = mylib.get_stock_prices("USDJPY=X")
    # 特徴量の計算
    df_USDJPY = mylib.colculate_feature(df_USDJPY, objective=False, exclude=exclude_feature)
    # 整形
    df_USDJPY = mylib.shaping_yfinance(df_USDJPY, begin=begin, end=end, drop_columns=["Volume", "Dividends", "Stock Splits"])
    # 欠損値がある行の削除
    df_USDJPY.dropna(subset=(set(feature) - set(exclude_feature)), inplace=True)
    # カラム名の変更
    [df_USDJPY.rename(columns={columns: "USDJPY_" + columns}, inplace=True) for columns in df_USDJPY.columns]


    # 結果を所持する辞書
    result = {
        "params": {},
        "feature importance": {},
        "mae": {},                  # 平均絶対誤差 = 1/N * Σ |yi - fi|                              | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数
        "mse": {},                  # 平均二乗誤差 = 1/N * Σ (yi - fi)^2                            | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数
        "rmse": {},                 # 平均二乗偏差 = √(1/N * Σ (yi - fi)^2)                         | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数
        "mape": {},                 # 平均絶対誤差率 = 1/N * Σ |(yi - fi) / yi|                     | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数
        "smape": {},                # 平均パーセント誤差 = 100/N * Σ | 2(yi - fi) / (|yi| + |fi|) | | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数
        "r^2": {}                   # R^2 = 1 - Σ ((yi - fi)^2 / (yi - ý))                          | yi:i番目の実測値, fi:i番目の予測値, ý:実測値の平均値
    }

    # 銘柄群に対して実行
    for security_code in security_codes:

        # 株価データフレームの作成
        # データのダウンロード
        # mylib.stock_prices_to_csv(security_code)
        # 取得したデータの読み取り
        df = mylib.get_stock_prices(security_code)
        # 欠損値がある行の削除
        df.dropna(inplace=True)
        # 特徴量の計算
        df = mylib.colculate_feature(df, objective="regression")
        # データの整形
        df = mylib.shaping_yfinance(df, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"] + drop_feature)
        # 株価指標データの結合
        # df = pd.concat([df, df_N225, df_DJI, df_GSPC, df_USDJPY], axis=1)
        df = pd.concat([df, df_DJI, df_USDJPY], axis=1)
        # 欠損値がある行の削除
        df.dropna(subset=(feature + ["target"]), inplace=True)


        # 学習データ、テストデータの作成
        # train 学習時データ test 学習後のテストデータ
        df_X = df.drop(["target"], axis=1)
        df_y = (df["target"].copy().astype(np.float64))

        # 1点で分割
        X_train = df_X[df_X.index <= test_begin]
        X_test = df_X[df_X.index > test_begin]
        y_train = df_y[df_y.index <= test_begin]
        y_test = df_y[df_y.index > test_begin]
        Xy_test = df[df.index > test_begin]


        # ハイパーパラメータの取得
        opt = optuna.create_study()
        opt.optimize(Objective(df_X, df_y, seed=seed), n_trials=31)
        result["params"][security_code] = opt.best_params
        result["params"][security_code].update({"num_iterations": 1000})


        # K-分割交差検証法(k-fold cross-validation)を行うためのモデル作成
        models = []
        y_preds = []
        result["mae"][security_code] = []
        result["mse"][security_code] = []
        result["rmse"][security_code] = []
        result["mape"][security_code] = []
        result["smape"][security_code] = []
        result["r^2"][security_code] = []
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
            model = lgb.train(params=result["params"][security_code], train_set=lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=False)

            # モデルの保存
            models.append(model)

            # 特徴量の重要度の保存
            result["feature importance"][security_code] += (model.feature_importance() / kfold_splits)

            # テストデータの予測
            y_preds.append(model.predict(X_test))

            # 訓練結果の評価
            # 平均絶対誤差(Mean Absolute Error)
            result["mae"][security_code].append(metrics.mean_absolute_error(y_test, y_preds[-1]))
            # 平均二乗誤差(Mean Squared Error)
            result["mse"][security_code].append(metrics.mean_squared_error(y_test, y_preds[-1]))
            # 平均二乗偏差(Root Mean Squared Error)
            result["rmse"][security_code].append(np.sqrt(metrics.mean_squared_error(y_test, y_preds[-1])))
            # 平均絶対誤差率(Mean Absolute Percentage Error)
            result["mape"][security_code].append(metrics.mean_absolute_percentage_error(y_test, y_preds[-1]))
            # 平均パーセント誤差(Symmetric Mean Absolute Percentage Error)
            result["smape"][security_code].append(100/len(y_test)*np.sum(2*np.abs(y_test - y_preds[-1]) / (np.abs(y_test) + np.abs(y_preds[-1]))))
            # R^2(決定係数:coefficient of determination)
            result["r^2"][security_code].append(metrics.r2_score(y_test, y_preds[-1]))


        # 株価の変動予測
        Xy_test.insert(len(Xy_test.columns), "predict", np.mean(y_preds, axis=0))
        Xy_test.insert(len(Xy_test.columns), "isbuy", (Xy_test["predict"].copy() >= isbuy_threshold))
        # 予測結果の保存
        mylib.isbuy_dataset_to_csv(Xy_test, security_code)


        """
        # Protra変換部分
        trading_days = {security_code: Xy_test[Xy_test["isbuy"] == True]}
        mylib.conversion_to_protra(trading_days, os.path.relpath(__file__))
        """


    """
    # 特徴量の重みを描画
    # 平均の計算
    for feature_importance in result["feature importance"].values():
        # ソート
        feature_importance.sort_values(ascending=True, inplace=True)
        # 出力
        print(feature_importance)
        # グラフ描画
        plt.figure(figsize=(10.24, 7.68))
        plt.barh(feature_importance.index, feature_importance)
        plt.title("regression model's feature important")
        #plt.grid(False)
        plt.show()
        plt.close()
    """


    print(result)

    # 評価関数
    def print_average_score(index):
        for security_code in security_codes:
            print("    {}: {}".format(security_code, np.lib.average(result[index][security_code])))
    # 平均スコアの出力
    # 平均絶対誤差(Mean Absolute Error)
    print("平均絶対誤差 = 1/N * Σ |yi - fi|  | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数")
    print_average_score("mae")
    # 平均二乗誤差(Mean Squared Error)
    print("平均二乗誤差 = 1/N * Σ (yi - fi)^2  | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数")
    print_average_score("mse")
    # 平均二乗偏差(Root Mean Squared Error)
    print("平均二乗偏差 = √(1/N * Σ (yi - fi)^2)  | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数")
    print_average_score("rmse")
    # 平均絶対誤差率(Mean Absolute Percentage Error)
    print("平均絶対誤差率 = 1/N * Σ |(yi - fi) / yi|  | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数")
    print_average_score("mape")
    # 平均パーセント誤差(Symmetric Mean Absolute Percentage Error)
    print("平均パーセント誤差 = 100/N * Σ | 2(yi - fi) / (|yi| + |fi|) |  | yi:i番目の実測値, fi:i番目の予測値, N:サンプル数")
    print_average_score("smape")
    # R^2(決定係数:coefficient of determination)
    print("R^2 = 1 - Σ ((yi - fi)^2 / (yi - ý))  | yi:i番目の実測値, fi:i番目の予測値, ý:実測値の平均値")
    print_average_score("r^2")


if __name__=="__main__":
    main()
