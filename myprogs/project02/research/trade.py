# データセットを元に取引結果を算出

# モジュールのインポート
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import datetime
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mylibrary as mylib


class Trade:
    def __init__(self, df_pred, dfs, security_codes, position=1e7, is_seles_commision=True,cut_loss_line=1e-1,  is_taxation=True, tax_rate=0.2):
        """変数の初期化
        インスタンス作成時に一度だけ実行
        """
        self.df_pred = df_pred
        self.dfs = dfs
        self.security_codes = security_codes
        self.position = position
        self.is_seles_commision = is_seles_commision
        self.cut_loss_line = 1 - cut_loss_line
        self.is_taxation = is_taxation
        self.tax_rate = tax_rate
        self.trade_num = {"sum": 0}
        for security_code in self.security_codes:
            self.trade_num[security_code] = 0
        self.cutloss1_num = {"sum": 0}
        for security_code in self.security_codes:
            self.cutloss1_num[security_code] = 0
        self.cutloss2_num = {"sum": 0}
        for security_code in self.security_codes:
            self.cutloss2_num[security_code] = 0

    def __call__(self):
        """取引の実施
        オブジェクトが呼び出されたときに実行
        """
        isbuy = False
        # 保有株式の記憶先
        stocks = [{"ticker": self.security_codes[0], "quantity": 0, "price":0}] * 3

        for index, row in self.df_pred.iterrows():
            # 3日前の株の売却
            stock = stocks.pop(0)
            ticker = stock["ticker"]
            num = stock["quantity"]
            if num:
                self.position += self.settlement_amout(self.dfs[ticker].at[index, "Open"].copy(), num)

            # 本日購入する株式の情報
            today = {"ticker": "", "quantity": 0, "price":0}
            # 購入する銘柄を取得
            today["ticker"] = row["whatbuy"]
            # 購入部分
            if isbuy:
                today["quantity"] = 1000000 // self.dfs[today["ticker"]].at[index, "Open"].copy()
                today["price"] = self.dfs[today["ticker"]].at[index, "Open"].copy()
                self.position -= self.trade_amount(today["price"], today["quantity"], bs=False)
                self.trade_num["sum"] += 1
                self.trade_num[today["ticker"]] += 1
            # 株式の購入情報の記憶
            stocks.append(today)

            # 損切判断
            for i, stock in enumerate(stocks):
                if stock["quantity"]:
                    # 購入価格より1割以上の減少が見られた場合に損切
                    if stock["price"]*self.cut_loss_line > self.dfs[stock["ticker"]].at[index, "Open"].copy():
                        self.position += self.settlement_amout(self.dfs[stock["ticker"]].at[index, "Open"].copy(), stock["quantity"])
                        stocks[i]["quantity"] = 0
                        self.cutloss1_num["sum"] += 1
                        self.cutloss1_num[stock["ticker"]] += 1
                    elif stock["price"]*self.cut_loss_line > self.dfs[stock["ticker"]].at[index, "Low"].copy():
                        self.position += self.settlement_amout(stock["price"]*self.cut_loss_line, stock["quantity"])
                        stocks[i]["quantity"] = 0
                        self.cutloss2_num["sum"] += 1
                        self.cutloss2_num[stock["ticker"]] += 1

            # 明日の株式の購入の是非を取得
            isbuy = row["hm_buy"]
            #isbuy = True
            #isbuy = random.randrange(2)

            # 資産状況を元DataFrameに貼り付け
            self.df_pred.at[index, "position"] = self.position
            self.df_pred.at[index, "market value"] = self.position
            for stock in stocks:
                self.df_pred.at[index, "market value"] += self.market_value([stock], self.dfs[stock["ticker"]].at[index, "Close"].copy())
            self.df_pred.at[index, "book value"] = self.position + self.book_value(stocks)

            # 所持株式の情報を保存
            for i, stock in enumerate(stocks):
                self.df_pred.at[index, "ticker(" + str(i) + ")"] = stocks[i]["ticker"]
                self.df_pred.at[index, "quantity(" + str(i) + ")"] = stocks[i]["quantity"]
                self.df_pred.at[index, "price(" + str(i) + ")"] = stocks[i]["price"]

        # 保有株式の売却
        for i, stock in enumerate(stocks):
            self.position += self.settlement_amout(self.dfs[stock["ticker"]].iat[len(self.dfs) - 1, 3].copy(), stock["quantity"])
            self.df_pred.at[index, "quantity(" + str(i) + ")"] = 0

    def settlement_amout(self, price, quantity):
        """受渡金額の計算

        Args:
            price (double): 約定時の株価
            quantity (int): 株式数

        Returns:
            [double]: 受渡金額
        """
        trade_amount = self.trade_amount(price, quantity, bs=True)
        # 手数料の支払い
        if self.is_seles_commision:
            trade_amount -=self.seles_commision(trade_amount)
        # 課税
        if self.is_taxation:
            trade_amount -= self.taxation_on_capital_gain(trade_amount)
        return trade_amount

    def trade_amount(self, price, quantity, bs=True):
        """約定金額の計算

        Args:
            price (double): 約定時の株価
            quantity (int): 株式数
            bs (bool or str): 売り買い
                True or "Sell": 売り
                False or "Buy": 買い

        Returns:
            [int]: 約定金額
        """
        # bsの内容の確認
        if isinstance(bs, str):
            bs = bs.capitalize()
            if bs == "Buy":
                bs = False
            elif bs == "Sell":
                bs == True
            else:
                raise ValueError(
                    "bs must be 'Buy' or 'Sell' (Not case-sensitive) ."
                )
        elif not isinstance(bs, bool):
            raise ValueError(
                "objective type of 'bs' must be bool or str."
            )
        trade_amount = price * quantity
        if bs:
            return math.ceil(trade_amount)
        else:
            return math.floor(trade_amount)

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

    def taxation_on_capital_gain(self, trade_amount):
        """譲渡益課税の計算

        Args:
            trade_amount (double): 約定金額

        Returns:
            [double]: 譲渡益課税
        """
        trade_amount *= (self.tax_rate + 0.00315)
        return self.tax_rate

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

    def get_df_pred(self):
        """データを返すメソッド

        Returns:
            [pandas.DataFrame]: データフレーム
        """
        return self.df_pred

    def get_trade_num(self):
        """取引件数を返すメソッド

        Returns:
            [dict]: 取引件数
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
    security_codes = [
        # 時価総額上位10株
        "4063.T", "6098.T", "6861.T", "6758.T", "7203.T",
        "8035.T", "8306.T", "9432.T", "9433.T", "9984.T",
        # 時価総額上位20株
        "4519.T", "4661.T", "6367.T", "6501.T", "6594.T",
        "6902.T", "7741.T", "7974.T", "9983.T",
        # 日経平均
        "^N225"
    ]

    # グラフ描画用の辞書
    plot = {}
    # 予測値と銘柄購入判断を記憶するDataFrame
    df_pred = pd.DataFrame()
    # 各銘柄のデータを記憶する辞書
    # key: str, value: DataFrame
    dfs = {}

    for security_code in security_codes:
        # データセットの読み取り
        df = mylib.get_isbuy_dataset(security_code)

        # 予測値の記憶
        pred = df["predict"].copy().where(df["predict"].copy() > 0.5, 0.0).rename(security_code)
        df_pred = pd.concat([df_pred, pred], axis=1)

        # データの記憶
        dfs[security_code] = df

    # indexの修正
    df_pred.index = pd.to_datetime(df_pred.index)
    # 欠損値の削除
    df_pred.dropna(axis=0, inplace=True)
    # どの銘柄を購入するかを判断
    df_pred["hm_buy"] = df_pred.copy().max(axis=1)
    df_pred["whatbuy"] = df_pred.copy().idxmax(axis=1)


    trade = Trade(df_pred, dfs, security_codes, cut_loss_line=0.1)
    trade()

    print(trade.get_df_pred())
    #print(trade.get_df_pred().loc[datetime.datetime(*[2019, 10, 1]):datetime.datetime(*[2019, 10, 4])])
    print("取引回数: {}".format(trade.get_trade_num()["sum"]))
    print("損切(始値による)回数: {}".format(trade.get_cutloss1_num()["sum"]))
    print("損切(価格変動による)回数: {}".format(trade.get_cutloss2_num()["sum"]))

    mylib.plot_chart({
        "market value": trade.get_df_pred()["market value"],
        "book value": trade.get_df_pred()["book value"],
    })

    return 0


if __name__=="__main__":
    main()
