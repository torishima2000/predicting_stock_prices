# 指定データを取得するモジュール群

# モジュールのインポート
import os
import json
import pandas as pd

# 設定ファイルの読み込み
path_name = {}
with open("mylibrary\path.json", "r") as f:
    path_name = json.load(f)

def codelist_topix500():
    """TOPIX500構成銘柄の証券コードリストを取得

    Returns:
        [list]: TOPIX500構成銘柄の証券コード
    """
    # TOPIX500構成銘柄情報をcsvファイルから取得
    path = os.path.join(path_name["TSE_listed_Issues"], path_name["TOPIX500"] + ".csv")
    list_topix500 = pd.read_csv(path)

    # 証券コード部分のみ摘出
    codes = list_topix500["Local Code"].values.tolist()
    return codes
