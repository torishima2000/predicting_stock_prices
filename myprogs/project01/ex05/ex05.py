# 自作モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import mylibrary as mylib

# 銘柄のサマリー
Security_code = "7203.T"
mylib.sammary_to_csv(Security_code)
my_info = mylib.get_sammary(Security_code)
print(my_info)
