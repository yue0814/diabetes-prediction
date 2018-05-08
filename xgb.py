# -*- coding: utf-8 -*-
"""
Created on May 3, 2018
@author: Yue Peng
"""
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import time
import _data


train_data, test_data = _data.data()


# =================== train ============================ #
# columns used to train
cols = train_data.columns.values.tolist()
# remove label
cols.remove("diabetes")


print("CV 5-fold train begin...")
t0 = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2018)
scores = []
for i, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print("The {0} round train...".format(i + 1))
    model_xgb = xgb.XGBClassifier(max_depth=6, gamma=0.2, colsample_bytree=0.6,
                                  min_child_weight=12, learning_rate=0.02,
                                  objective="binary:logistic",
                                  silent=1, eval_metric="auc")
    train_feat1 = train_data[cols].iloc[train_idx, :]
    train_feat2 = train_data[cols].iloc[val_idx, :]
    train_target1 = train_data.diabetes.iloc[train_idx]
    train_target2 = train_data.diabetes.iloc[val_idx]
    model_xgb.fit(train_feat1, train_target1)
    print('Train auc', roc_auc_score(train_target1, model_xgb.predict_proba(train_feat1)[:, 1]))
    print('Test auc', roc_auc_score(train_target2, model_xgb.predict_proba(train_feat2)[:, 1]))
    scores.append(roc_auc_score(train_target2, model_xgb.predict_proba(train_feat2)[:, 1]))

print("The average test auc is {0}".format(np.mean(scores)))


model_xgb.fit(train_data[cols], train_data["diabetes"])
roc_auc_score(test_data.diabetes, model_xgb.predict_proba(test_data[cols])[:, 1])
