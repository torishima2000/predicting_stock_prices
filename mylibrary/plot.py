# グラフ描画を行うモジュール群

# モジュールのインポート
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_chart(data, size={"x":10.24, "y":7.68}):
    fig = plt.figure(figsize=(size["x"], size["y"]))
    for k, v in data.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.xlabel("date")
    plt.ylabel("price")
    plt.show()
    plt.close()
