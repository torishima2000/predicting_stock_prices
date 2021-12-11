# 2値分類モデルのスコアを算出する

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
from sklearn import metrics
import mylibrary as mylib



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
print("正解率:{}".format(metrics.accuracy_score(y_true=y_test, y_pred=(y_pred > 0.5))))
print("適合率:{}".format(metrics.precision_score(y_true=y_test, y_pred=(y_pred > 0.5))))
print("再現率:{}".format(metrics.recall_score(y_true=y_test, y_pred=(y_pred > 0.5))))
print("F値: {}".format(metrics.f1_score(y_true=y_test, y_pred=(y_pred > 0.5))))
print("AUC: {}".format(metrics.roc_auc_score(y_true=y_test, y_score=y_pred)))


f = {
    "SMA3": 0.4686937175548258,
    "SMA5": 0.42725473565168226,
    "SMA15": 0.45571965774701617,
    "SMA25": 0.4906870229007633,
    "SMA50": 0.49803861633297675,
    "SMA75": 0.504436378302077,
    "SMA100": 0.526055072604536,
    "WMA5": 0.44891488547531844,
    "upper1": 0.4686112872542803,
    "lower1": 0.4754716981132075,
    "upper2": 0.45864045864045866,
    "lower2": 0.47582388610598464,
    "upper3": 0.4454298542209412,
    "lower3": 0.5066933066933066,
    "MACD": 0.4028794098399666,
    "MACDsignal": 0.39137283412092577,
    "MACDhist": 0.45430470734475487,
    "RSI9": 0.43718648980046815,
    "RSI14": 0.4278590048736257,
    "VR": 0.4965796807702052,
    "MAER15": 0.4422093798621916,
    "ADX": 0.4422093798621916,
    "CCI": 0.4478332419089413,
    "ROC": 0.48733525535420097,
    "ADOSC": 0.4603303547251557,
    "ATR": 0.4210885514417335,
    "DoD1": 0.5137551917908625,
    "DoD2": 0.4979156075241484,
    "DoD3": 0.4508967629046369,
}

auc = {
    "SMA3": 0.5019452986854772,
    "SMA5": 0.5004762669759277,
    "SMA15": 0.4992400750970538,
    "SMA25": 0.5042069953899619,
    "SMA50": 0.5064222224499499,
    "SMA75": 0.5059766683157334,
    "SMA100": 0.5083395805584712,
    "WMA5": 0.5001210268415945,
    "upper1": 0.501247453254168,
    "lower1": 0.504347032771038,
    "upper2": 0.5081628398111125,
    "lower2": 0.5094312244244006,
    "upper3": 0.5010178114108059,
    "lower3": 0.5055544376959648,
    "MACD": 0.5087476406987967,
    "MACDsignal": 0.5102547137598124,
    "MACDhist": 0.49795044996897847,
    "RSI9": 0.5010003871946669,
    "RSI14": 0.5010363607508492,
    "VR": 0.5066040262548778,
    "MAER15": 0.5049319755662416,
    "ADX": 0.5049319755662416,
    "CCI": 0.5050952604398802,
    "ROC": 0.504411818621796,
    "ADOSC": 0.5062217324655851,
    "ATR": 0.49242668456971933,
    "DoD1": 0.5002984213771677,
    "DoD2": 0.5062757434811158,
    "DoD3": 0.5040447849589706,
}

print("f値")
li = sorted(f.items(), key=lambda x:x[1])
for k, v in li:
    print(k + ":" + str(v))
print("AUC")
li = sorted(auc.items(), key=lambda x:x[1])
for k, v in li:
    print(k + ":" + str(v))
