from time import time

import numpy as np
from pyod.utils import precision_n_scores
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import check_consistent_length, column_or_1d


def evaluate_clf(mat_file, clf_name, y, y_pred, training_time, testing_time):
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)

    f1 = np.round(f1_score(y, y_pred), decimals=4)
    prc = np.round(precision_score(y, y_pred), decimals=4)
    recall = np.round(recall_score(y, y_pred), decimals=4)
    mcc = np.round(matthews_corrcoef(y, y_pred), decimals=4)

    new_row = {
        "dataset": mat_file,
        "model": clf_name,
        "prc": prc,
        "recall": recall,
        "mcc": mcc,
        "f1": f1,
        "training_time": training_time,
        "testing_time": testing_time,
    }

    return new_row
