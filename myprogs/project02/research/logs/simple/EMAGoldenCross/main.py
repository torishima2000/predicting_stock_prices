# 二値分類モデル

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import pandas as pd
import mylibrary as mylib


def main():
    # 変数の定義
    # 証券コード
    security_codes = [
        #2021/12/4
        # 時価総額上位10株
        "7203.T", "6758.T", "6861.T", "6098.T", "9432.T",
        "9984.T", "8035.T", "8306.T", "4063.T", "6594.T",
        # 時価総額上位20株
        "9433.T", "6367.T", "9983.T", "6902.T",
        "7741.T", "6501.T", "7974.T", "4661.T", "4519.T",
        # 日経平均
        "^N225"
    ]
    #security_codes = []

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
        "upper1", "lower1", "upper2", "lower2", "upper3", "lower3",
        "MACD", "MACDsignal", "MACDhist", "MACDGoldenCross",
        "RSI9", "RSI14",
        "VR", "MAER15",
        "ADX", "CCI", "ROC", "ADOSC", "ATR",
        "DoD1", "DoD2", "DoD3",
    ]
    # 削除する特徴量
    drop_feature = [
    ]
    # 特徴量カラムの修正
    for v in drop_feature:
        feature.remove(v)
    # 追加の特徴量
    add_feature = ["N225_" + column for column in feature]
    add_feature += ["DJI_" + column for column in feature]
    add_feature += ["GSPC_" + column for column in feature]

    # 買い判断をするための閾値
    isbuy_threshold = 0.5

    # 結果を所持するDataFrame
    result = {
        "params": {},
        "feature importance": {},
        "accuracy": {},             # 正解率 = (TP + TN) / (TP + FP + FN + TP)
        "precision": {},            # 適合率 = TP / (TP + FP)
        "recall": {},               # 再現率 = TP / (TP + FN)
        "f": {},                    # F値 = 2*Precision*Recall / (Precision + Recall)
        "Log loss": {},
        "AUC": {}
    }

    # 銘柄群に対して実行
    for i, security_code in enumerate(security_codes):

        # 株価データフレームの作成
        # データのダウンロード
        # mylib.stock_prices_to_csv(security_code)
        # 取得したデータの読み取り
        df = mylib.get_stock_prices(security_code)
        # 欠損値がある行の削除
        df.dropna(inplace=True)
        # 特徴量の計算
        df = mylib.colculate_feature(df, objective="binary")
        # データの整形
        df = mylib.shaping_yfinance(df, begin=begin, end=end, drop_columns=["Dividends", "Stock Splits"] + drop_feature)
        # 欠損値がある行の削除
        df.dropna(subset=(feature + ["target"]), inplace=True)


        # 学習データ、テストデータの作成
        # train 学習時データ test 学習後のテストデータ
        df_X = df.drop(["growth rate", "target"], axis=1)
        df_y = (df["target"].copy().astype(bool))

        # 1点で分割
        X_train = df_X[df_X.index <= test_begin]
        X_test = df_X[df_X.index > test_begin]
        y_train = df_y[df_y.index <= test_begin]
        y_test = df_y[df_y.index > test_begin]
        Xy_test = df[df.index > test_begin]

        # ゴールデンクロスによる単純な取引戦略
        # 株価の変動予測
        Xy_test.insert(len(Xy_test.columns), "predict", (Xy_test["EMAGoldenCross"].copy() - (1e-3 * i)))
        Xy_test.insert(len(Xy_test.columns), "isbuy", (Xy_test["predict"].copy() >= isbuy_threshold))
        # 予測結果の保存
        mylib.isbuy_dataset_to_csv(Xy_test, security_code)


if __name__=="__main__":
    main()
