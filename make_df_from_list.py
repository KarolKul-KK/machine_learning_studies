import pandas as pd

import os


x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
lists = [x, y1, y2, y3, x4, y4]


def make_df_from_lists(lists: list) -> pd.DataFrame:

    df = pd.DataFrame(lists, columns=lists)

    return df


def metrics(df: pd.DataFrame) -> pd.DataFrame:

    df_mean = df.mean().round(2)
    df_std = df.std(ddof=0).round(2)
    df_var = df.var().round(2)

    df_final = pd.DataFrame([df_mean, df_std, df_var]).T
    df_final.columns = ['Mean', 'Std', 'Var']

    return df_final


def make_csv_from_df(df: pd.DataFrame, outpath: str) -> None:

    df.to_csv(os.path.join(outpath, 'final.csv'))


output_path = '/Users/karolkul/Documents/GitHub/machine_learning_studies/machine_learning_studies'
df = make_df_from_lists(lists)
df_final = metrics(df)
make_csv_from_df(df_final, output_path)


