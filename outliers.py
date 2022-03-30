import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


df = pd.read_csv('houses_data.csv', usecols=['Price', 'Bathroom', 'Landsize', 'Propertycount'])

def zscore_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    columns = df.columns
    df_zscore = (df - df.mean())/df.std(ddof=0)

    for column in columns:
        df_new = df.where(df_zscore[f'{column}'] < 2).dropna().reset_index(drop=True)

    return df_new, (len(df) - len(df_new))


def m_zscore_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    columns = df.columns
    df_zscore = (df - df.mean())/df.std(ddof=0).abs()

    for column in columns:
        df_new = df.where(df_zscore[f'{column}'] < 3.5).dropna().reset_index(drop=True)

    return df_new, (len(df) - len(df_new))


def quantile_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    df_new = df.copy()
    q_min = df.quantile(q=0.1)
    q_max =df.quantile(q=0.9)

    for i in range(len(df)):
        if (df.iloc[i] > q_max).any():
            df_new.drop(i, inplace=True)
        elif (df.iloc[i] < q_min).any():
            df_new.drop(i, inplace=True)
        else:
            pass
        
    return df_new.reset_index().drop('index', axis=1), (len(df) - len(df_new))