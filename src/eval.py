from typing import Callable, Dict

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _score(score: Callable[[pd.DataFrame, pd.DataFrame], float], true: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    result = {'all': score(true, pred, average='micro')}
    for col in true.columns:
        result[col] = score(true[col], pred[col])
    return result


def f1(true: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    return _score(f1_score, true, pred)


def precision(true: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    return _score(precision_score, true, pred)


def recall(true: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
    return _score(recall_score, true, pred)
