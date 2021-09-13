# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import mylibrary as mylib

# 銘柄リストの読み込み
my_data = mylib.get_codelist_topix500()
print(my_data)
