# 計量的・実証的トレーディングの実行
# 銘柄リストの読み込み
# https://qiita.com/blog_UKI/items/f782fb86747e0bae89a9#%E9%8A%98%E6%9F%84%E3%83%AA%E3%82%B9%E3%83%88%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")
print(data)
