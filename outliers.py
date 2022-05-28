import pandas as pd
import numpy as np
from typing import Tuple

from missing_values import data_split, score_dataset


def read_csv() -> pd.DataFrame:

    df = pd.read_csv(
        "houses_data.csv",
        usecols=["Price", "Bathroom", "Landsize", "Propertycount", "Postcode"],
        dtype={
            "Price": float,
            "Bathroom": int,
            "Landsize": float,
            "Propertycount": float,
            "Postcode": float,
        },
    )

    return df


def zscore_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    columns = df.columns
    df_zscore = (df - df.mean()) / df.std(ddof=0)

    for column in columns:
        df_new = df.where(df_zscore[f"{column}"] < 2).dropna().reset_index(drop=True)

    return df_new, (len(df) - len(df_new))


def m_zscore_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    columns = df.columns
    df_zscore = (df - df.mean()) / df.std(ddof=0).abs()

    for column in columns:
        df_new = df.where(df_zscore[f"{column}"] < 3.5).dropna().reset_index(drop=True)

    return df_new, (len(df) - len(df_new))


def quantile_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    df_new = df.copy()
    q_min = df.quantile(q=0.1)
    q_max = df.quantile(q=0.9)

    for i in range(len(df)):
        if (df.iloc[i] > q_max).any():
            df_new.drop(i, inplace=True)
        elif (df.iloc[i] < q_min).any():
            df_new.drop(i, inplace=True)
        else:
            pass

    return df_new.reset_index().drop("index", axis=1), (len(df) - len(df_new))


def IQR_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    df_new = df.copy()
    Q1 = df.quantile(q=0.25)
    Q3 = df.quantile(q=0.75)
    IQR = Q1 - Q3

    for i in range(len(df)):
        if (
            (df.iloc[i] < (Q1 - 1.5 * IQR)) | (df.iloc[i] > (Q3 + 1.5 * IQR))
        ).all() == True:
            pass
        else:
            df_new.drop(i, inplace=True)

    return df_new.reset_index().drop("index", axis=1), (len(df) - len(df_new))


def main():

    df = read_csv()
    df_zscore, diff_zscore = zscore_outliers(df)
    df_m_zscore, diff_m_zscore = m_zscore_outliers(df)
    df_quantile, diff_quantile = quantile_outliers(df)
    df_iqr, diff_iqr = IQR_outliers(df)

    df_list = [df_zscore, df_m_zscore, df_quantile, df_iqr]
    diff_list = [diff_zscore, diff_m_zscore, diff_quantile, diff_iqr]

    for i in range(len(df_list)):
        X_train, X_test, y_train, y_test = data_split(df_list[i].dropna())
        mae = score_dataset(X_train, X_test, y_train, y_test)
        print({"mae": mae, "diff": diff_list[i]})


if __name__ == "__main__":
    main()
