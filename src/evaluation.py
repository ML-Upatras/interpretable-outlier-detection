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


def evaluate_clf(mat_file, clf_name, y, y_pred, y_proba, start_time):
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    y_proba = column_or_1d(y_proba)

    # TODO: Remove this after deciding the correct function
    # print("========================================")
    # print(y_pred)
    # print("========================================")
    # print(y_proba)
    # print("========================================")
    check_consistent_length(y, y_pred)

    roc = np.round(roc_auc_score(y, y_proba), decimals=4)
    prn_n = np.round(precision_n_scores(y, y_proba), decimals=4)

    f1 = np.round(f1_score(y, y_pred), decimals=4)
    prc = np.round(precision_score(y, y_pred), decimals=4)
    recall = np.round(recall_score(y, y_pred), decimals=4)
    mcc = np.round(matthews_corrcoef(y, y_pred), decimals=4)

    new_row = {
        "dataset": mat_file,
        "model": clf_name,
        "roc": roc,
        "prn_n": prn_n,
        "prc": prc,
        "recall": recall,
        "mcc": mcc,
        "f1": f1,
        "time": round(time() - start_time, 4),
    }

    return new_row
