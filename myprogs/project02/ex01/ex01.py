# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import mylibrary as mylib

# 株価データの取得
reference_ticker = "^N225"
N225 = mylib.get_stock_prices(reference_ticker)

# rsiの計算
N225["RIS9"] = talib.RSI(np.array(N225["Close"]), timeperiod=9)

plt.figure(figsize=(20.24, 7.68))
plt.plot(N225["RIS9"])
plt.show()
