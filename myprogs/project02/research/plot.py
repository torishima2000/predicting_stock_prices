# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import mylibrary as mylib

# モジュールのインポート
import datetime
import matplotlib.pyplot as plt

def comparison(li):
    plot = {}
    for v in li:
        df = mylib.get_isbuy_dataset(v)
        print(v)
        print(df.iloc[:, 20:25])
        plot[v] = df["market value"]
    plot_chart(plot)

def plot_chart(data, size={"x":10.24, "y":7.68}):
    fig = plt.figure(figsize=(size["x"], size["y"]))
    for k, v in data.items():
        plt.plot(v, label=k)
    plt.legend(fontsize=14, loc="lower left")
    plt.xlabel("date", fontsize=16)
    plt.ylabel("total assets(market value)", fontsize=16)
    plt.tick_params(labelsize=12)
    begin = datetime.datetime(*[2017, 1, 1])
    end = datetime.datetime(*[2021, 1, 1])
    plt.xlim(begin, end)
    plt.ylim(0.6e7, 1.1e7)
    plt.hlines(1e7, xmin=begin, xmax=end, color="#A0A0A0", linestyles="solid")
    plt.show()
    plt.close()

def main():
    # 指数平滑移動平均
    ema = [
        "EMA(50)",
        "EMA(75)",
        "EMA(25, 75, GC)",
        "only GoldenCross"
    ]

    # 単純移動平均
    sma = [
        "SMA(100)",
        "SMA(25, 75, GC)",
        "only GoldenCross"
    ]

    # 加重移動平均
    wma = [
        "WMA(75)",
        "WMA(25, 75, GC)",
        "only GoldenCross"
    ]

    comparison(ema)

if __name__=="__main__":
    main()

