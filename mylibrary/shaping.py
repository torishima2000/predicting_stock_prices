# yfinanceからダウンロードしたpandas.DataFrameの整形

# モジュールのインポート
import datetime
import pandas as pd


def shaping_yfinance(df, begin=datetime.date.min, end=datetime.date.max, drop_columns=[]):
    """yfinanceから取得したDataFrameの整形を行うメソッド

    Args:
        df (pd.DataFrame): yfinanceから取得したDataFrame
        begin (datetime, optional): データの開始日(この日付を含む). Defaults to datetime.date.min.
        end (datetime, optional): データの終了日(この日付を含む). Defaults to datetime.date.max.
        drop_columns (list, optional): 削除するカラムのリスト. Defaults to [].

    Returns:
        [type]: [description]
    """
    # 日付での整形
    df = df[df.index >= begin]
    df = df[df.index <= end]

    # 不要カラムの削除
    df.drop(drop_columns, axis=1, inplace=True)

    return df