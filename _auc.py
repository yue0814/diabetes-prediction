# -*- coding: utf-8 -*-
from sklearn.metrics import roc_auc_score

def auc(preds, data):
    return "auc", roc_auc_score(data.get_label().squeeze(), preds), False