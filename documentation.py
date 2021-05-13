# 資料作成用のプログラム


# 銘柄リストの読み込み
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv")
print(data)