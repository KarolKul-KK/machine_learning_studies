import pandas as pd
import numpy as np


def read_csv(filename: str) -> pd.core.frame.DataFrame:
    '''Getting only numeric columns from file.'''
    df = pd.read_csv(f'{filename}.csv')
    cols = df.columns
    num_cols = []
    
    for col in cols:
        if type(df[f'{col}'].iloc[0]) == np.float64:
            num_cols.append(col)
        else:
            pass

    df_num = pd.read_csv(f'{filename}.csv', usecols=num_cols)

    return df_num


def dropping_missing_values(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    return df.dropna()


def mean_fillinf_values(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    return df.fillna(df.mean())


def knn_filling(df: pd.core.frame.DataFrame) -> np.ndarray:

    impt = KNNImputer()
    impt.fit(df)
    impt_results = impt.transform(df)

    return impt_results
