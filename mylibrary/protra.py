# Protra関連のメソッド

# モジュールのインポート
import os
import sys
import json

# 設定ファイルの読み込み
path_name = {}
with open("mylibrary\path.json", "r") as f:
    path_name = json.load(f)

def write_date(code, dates):
    """取引日をprotra言語方式に書き出す関数

    Args:
        code (string): 証券コード
        dates (Datetimeindex): 購入日のみを抽出したデータセット

    Returns:
        [String]: 購入日をもとにした売買基準をprotra用に記述した文字列
    """
    s = "def IsBUYDATE\n"
    s += "  if ((int)Code == " + code + ")\n"
    s += "     if ( \\\n"
    for index, row in dates.iterrows():
        s += "(Year == " + str(index.year)
        s += " && Month == " + str(index.month)
        s += " && Day == " + str(index.day) + ") || \\\n"
    s += "         (Year == 3000))\n"
    s += "         return 1\n"
    s += "     else\n"
    s += "         return 0\n"
    s += "     end\n"
    s += "  end\n"
    s += "end\n"
    return s

def conversion_to_protra(code, dates, relpath):
    """取引日をprotra言語で記述されたlibraryに変換する関数

    Args:
        code (string): 証券コード
        dates (Datetimeindex): 購入日のみを抽出したデータセット
        relpath (os.path): プログラムの相対パス
    """
    # protra言語で書かれたソースコードの取得
    soucecode = "# {}\n".format(relpath)
    soucecode += write_date(code, dates)

    # libの更新
    with open(os.path.join(path_name["protra"], "lib", "LightGBM.pt"), mode="w") as f:
        f.write(soucecode)

    # libraryを保存
    file_name = "LightGBM_{}.pt".format(relpath.replace("\\", "-").replace(".py", ""))
    with open(os.path.join(path_name["protra"], "lib", "LGBM_List", file_name), mode="w") as f:
        f.write(soucecode)
