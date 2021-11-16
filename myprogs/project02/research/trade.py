# データセットを元に取引結果を算出

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mylibrary as mylib


class Trade:
    def __init__(self, df, cut_loss_line=1e-1, position=1e7, is_seles_commision=True):
        """変数の初期化
        インスタンス作成時に一度だけ実行
        """
        self.df = df
        self.cut_loss_line = 1 - cut_loss_line
        self.position = position
        self.is_seles_commision = is_seles_commision
        self.trade_num = 0
        self.cutloss1_num = 0
        self.cutloss2_num = 0
    
    def __call__(self):
        """取引の実施
        オブジェクトが呼び出されたときに実行
        """
        isbuy = False
        # 保有株式の記憶先
        stocks = [{"quantity": 0, "price":0}] * 3

        for index, row in self.df.iterrows():
            # 3日前の株価の売却
            num = stocks.pop(0)["quantity"]
            if num:
                self.position += self.settlement_amout(row["Open"], num)

            today = {"quantity": 0, "price":0}
            # 購入部分
            if isbuy:
                today["quantity"] = 100000 // row["Open"]
                today["price"] = row["Open"]
                self.position -= today["price"] * today["quantity"]
                self.trade_num += 1
            # 株式の購入情報の記憶
            stocks.append(today)

            # 損切判断
            for i, stock in enumerate(stocks):
                if stock["quantity"]:
                    # 購入価格より1割以上の減少が見られた場合に損切
                    if stock["price"]*self.cut_loss_line > row["Open"]:
                        self.position += self.settlement_amout(row["Open"], stock["quantity"])
                        stocks[i]["quantity"] = 0
                        self.cutloss1_num += 1
                    elif stock["price"]*self.cut_loss_line > row["Low"]:
                        self.position += self.settlement_amout(stock["price"]*self.cut_loss_line, stock["quantity"])
                        stocks[i]["quantity"] = 0
                        self.cutloss2_num += 1

            # 明日の株式の購入の是非を取得
            isbuy = row["isbuy"]

            # 資産状況を元DataFrameに貼り付け
            self.df.at[index, "position"] = self.position
            self.df.at[index, "market value"] = self.position + self.market_value(stocks, row["Close"])
            self.df.at[index, "book value"] = self.position + self.book_value(stocks)

            # 所持株式の情報を保存
            for i, stock in enumerate(stocks):
                self.df.at[index, "quantity(" + str(i) + ")"] = stocks[i]["quantity"]
                self.df.at[index, "price(" + str(i) + ")"] = stocks[i]["price"]

        # 保有株式の売却
        for i, stock in enumerate(stocks):
            self.position += self.settlement_amout(self.df.iat[len(self.df) - 1, 3]*self.cut_loss_line, stock["quantity"])
            self.df.at[index, "quantity(" + str(i) + ")"] = 0

    def settlement_amout(self, price, quantity):
        trade_amount = self.trade_amount(price, quantity)
        if self.is_seles_commision:
            return trade_amount - self.seles_commision(trade_amount)
        else:
            return trade_amount

    def trade_amount(self, price, quantity):
        """約定金額の計算

        Args:
            price (double): 約定時の株価
            quantity (int): 株式数

        Returns:
            [double]: 約定金額
        """
        return price * quantity

    def seles_commision(self, trade_amount):
        """取引手数料の計算

        Args:
            trade_amount (double): 約定金額

        Returns:
            [double]: 取引手数料
        """
        if trade_amount < 10000:
            return 55
        else:
            return trade_amount * 0.0055

    def market_value(self, stocks, price):
        """所持株式の時価の計算

        Args:
            stocks (dict): 株式保持数が記載された辞書
                stocks["price"]: 購入時の株価
                stocks["quantity"]: 所持している株式数
            price (double): 株価

        Returns:
            [double]: 所持株式の時価
        """
        sum = 0
        for stock in stocks:
            sum += price * stock["quantity"]
        return sum

    def book_value(self, stocks):
        """所持株式の簿価の計算

        Args:
            stocks (dict): 株式保持数が記載された辞書
                stocks["price"]: 購入時の株価
                stocks["quantity"]: 所持している株式数

        Returns:
            [double]: 所持株式の簿価
        """
        sum = 0
        for stock in stocks:
            sum += stock["price"] * stock["quantity"]
        return sum

    def get_df(self):
        """データを返すメソッド

        Returns:
            [pandas.DataFrame]: データフレーム
        """
        return self.df

    def get_trade_num(self):
        """取引件数を返すメソッド

        Returns:
            [int]: 取引件数
        """
        return self.trade_num

    def get_cutloss1_num(self):
        """始値による損切回数を返すメソッド

        Returns:
            [pandas.DataFrame]: 始値による損切回数
        """
        return self.cutloss1_num

    def get_cutloss2_num(self):
        """株価による損切回数を返すメソッド

        Returns:
            [pandas.DataFrame]: 株価による損切回数
        """
        return self.cutloss2_num


def main():
    # ログファイルの保存場所
    logfile = os.path.join("myprogs", "project02", "research", "logs")

    # 証券コード
    # security_code = "7203.T"
    security_codes = ["6758.T", "7203.T", "9984.T", "^N225"]


    for security_code in security_codes:
        # データセットの読み取り
        df = mylib.get_isbuy_dataset(security_code)

        # 取引部分
        trade = Trade(df)
        trade()
        pd.set_option("display.max_rows", None)
        print(trade.get_df())
        # 総資産のグラフの描画
        mylib.plot_chart({
#            security_code + "(cash)": trade.get_df()["position"],
            security_code + "(market value)": trade.get_df()["market value"],
#            security_code + "(book value)": trade.get_df()["book value"]
        })


if __name__=="__main__":
    main()
