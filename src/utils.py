import pandas as pd


def clear_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.replace({'obecny': True, 'brak': False})
    dataframe = dataframe.sort_index()
    dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
    return dataframe
