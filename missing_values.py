import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def read_csv(filename: str) -> pd.DataFrame:
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


def dropping_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    return df.dropna()


def mean_filling_values(df: pd.DataFrame) -> pd.DataFrame:

    return df.fillna(df.mean())


def knn_filling(df: pd.DataFrame) -> pd.DataFrame:

    columns = df.columns
    impt = KNNImputer()
    impt.fit(df)
    impt_results = impt.transform(df)

    return pd.DataFrame(impt_results, columns=columns)


def data_split(df: pd.DataFrame) -> pd.DataFrame:

    X_train, X_test, y_train, y_test = train_test_split(df.drop('Price', axis=1),
                                                        df['Price'],
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


def score_dataset(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> np.float64:

    regr_model = LinearRegression()
    regr_model.fit(X_train, y_train)
    preds = regr_model.predict(X_test)

    return mean_absolute_error(y_test, preds)


def main() -> dict:

    df = read_csv('houses_data')
    df_drop = dropping_missing_values(df)
    df_mean = mean_filling_values(df)
    df_knn = knn_filling(df)

    df_list = [df_drop, df_mean, df_knn]

    for df in df_list:
        X_train, X_test, y_train, y_test = data_split(df)
        mae = score_dataset(X_train, X_test, y_train, y_test)
        print({'mae': mae})


if __name__ == '__main__':
    main()

