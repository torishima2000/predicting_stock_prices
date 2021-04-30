# 指定データを取得するモジュール群

# モジュールのインポート
import os
import pandas as pd

# 設定ファイルのインポート
from . import settings

def codelist_topix500():
    """TOPIX500構成銘柄の証券コードリストを取得

    Returns:
        [list]: TOPIX500構成銘柄の証券コード
    """
    # TOPIX500構成銘柄情報をcsvファイルから取得
    path = os.path.join(settings.directory_names["TSE_listed_Issues"], settings.file_names["TOPIX500"] + ".csv")
    list_topix500 = pd.read_csv(path)

    # 証券コード部分のみ摘出
    codes = list_topix500["Local Code"].values.tolist()
    return codes
