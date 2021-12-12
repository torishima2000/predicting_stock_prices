# 2値分類モデルのスコアを算出する

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import pandas as pd
from sklearn import metrics
import mylibrary as mylib


def main():
    # 変数
    threshold = 0.5

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

    df = pd.DataFrame(columns=["predict", "target"])

    for security_code in security_codes:
        # データセットの読み取り
        df_temp = mylib.get_isbuy_dataset(security_code)
        df_temp.insert(len(df_temp.columns), "growth rate", (df_temp["Open"].pct_change(3).shift(-4)))
        df_temp.insert(len(df_temp.columns), "target", (df_temp["growth rate"].copy() > 0))
        df = pd.concat([df, df_temp.loc[:, ["predict", "target"]]], axis=0)

    y_test = np.array(df["target"].copy().astype(bool))
    y_pred = np.array(df["predict"].copy())
    print("正解率:{}".format(metrics.accuracy_score(y_true=y_test, y_pred=(y_pred > threshold))))
    print("適合率:{}".format(metrics.precision_score(y_true=y_test, y_pred=(y_pred > threshold))))
    print("再現率:{}".format(metrics.recall_score(y_true=y_test, y_pred=(y_pred > threshold))))
    print("F値: {}".format(metrics.f1_score(y_true=y_test, y_pred=(y_pred > threshold))))
    print("ROC AUC: {}".format(metrics.roc_auc_score(y_true=y_test, y_score=y_pred)))
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    print("PR AUC: {}".format(metrics.auc(recall, precision)))


if __name__=="__main__":
    main()
