import pandas as pd
import numpy as np


def read_csv(filename):
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