# Protra関連のモジュール群

# モジュールのインポート
import os
import sys
import json

# 設定ファイルの読み込み
path_name = {}
with open("mylibrary\path.json", "r") as f:
    path_name = json.load(f)


def write_date(code, dates):
    """1つの銘柄の取引日をprotra言語方式に書き出す関数

    Args:
        code (string): 証券コード
        dates (Datetimeindex): 購入日のみを抽出したデータセット

    Returns:
        [type]: [description]
    """
    s = "  if ((int)Code == " + code + ")\n"
    s += "    if ( \\\n"
    for index, row in dates.iterrows():
        s += "      (Year == " + str(index.year)
        s += " && Month == " + str(index.month)
        s += " && Day == " + str(index.day) + ") || \\\n"
    s += "      (1 == 0))\n"
    s += "      return 1\n"
    s += "    end\n"
    s += "  end\n"
    return s

def write_code(trading_days):
    """銘柄とその取引日をprotra言語方式に書き出す関数

    Args:
        trading_days (dict): key(str) 証券コード, value(Datetimeindex) 購入日のみを抽出したデータセット

    Returns:
        [String]: 購入日をもとにした売買基準をprotra用に記述した文字列
    """
    s = "def IsBUYDATE\n"
    for code, dates in trading_days.items():
        s += write_date(code, dates)
    s += "  return 0\n"
    s += "end\n"
    return s

def conversion_to_protra(trading_days: dict, relpath):
    """取引日をprotra言語で記述されたlibraryに変換する関数

    Args:
        trading_days (dict): key(str) 証券コード, value(Datetimeindex) 購入日のみを抽出したデータセット
        relpath (os.path): プログラムの相対パス
    """
    # protra言語で書かれたソースコードの取得
    soucecode = "# {}\n".format(relpath)
    soucecode += write_code(trading_days)

    # libの更新
    with open(os.path.join(path_name["protra"], "lib", "LightGBM.pt"), mode="w") as f:
        f.write(soucecode)

    # libraryを保存
    file_name = "LightGBM_{}.pt".format(relpath.replace("\\", "-").replace(".py", ""))
    with open(os.path.join(path_name["protra"], "lib", "LGBM_List", file_name), mode="w") as f:
        f.write(soucecode)
