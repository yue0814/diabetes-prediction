# -*- coding: utf-8 -*-
"""
Created on May 3, 2018
@author: Yue Peng
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import catboost
from catboost import Pool
from sklearn.preprocessing import scale
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import time

import _data
from _auc import auc
import _variables

train_data, test_data = _data.data()


# =================== train ============================ #
# columns used to train
cols = train_data.columns.values.tolist()
# remove label
cols.remove("diabetes")


cat_feature_inds = []
for i, c in enumerate(train_data[cols].columns.values):
    num_uniques = len(train_data[cols][c].unique())
    if num_uniques < 5:
        cat_feature_inds.append(i)

print("CV 5-fold train begin...")
t0 = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2018)
scores = []
for i, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print("The {0} round train...".format(i + 1))
    # ---- lightgbm ------- #
    train_dat = lgb.Dataset(train_data[cols].iloc[train_idx], train_data["diabetes"].iloc[train_idx])
    valid_dat = lgb.Dataset(train_data[cols].iloc[val_idx], train_data["diabetes"].iloc[val_idx])
    gbm = lgb.train(_variables.lgb_params,
                    train_dat,
                    num_boost_round=_variables.num_boost_round,
                    valid_sets=valid_dat,
                    verbose_eval=500,
                    feval=auc)
    tree_feature_train = gbm.predict(train_data[cols].iloc[train_idx, :],
                                     pred_leaf=True)
    val_feature = gbm.predict(train_data[cols].iloc[val_idx, :],
                                     pred_leaf=True)
    # ----- catboost ----- #
    cat_model = catboost.CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=1,
        eval_metric='F1',
        random_seed=i * 100 + 6,
        logging_level="Silent"
    )
    # add tree features
    train_feat1 = np.concatenate((train_data.iloc[train_idx, :].as_matrix(cols),
                                  tree_feature_train), axis=1)
    train_feat2 = np.concatenate((train_data.iloc[val_idx, :].as_matrix(cols),
                                  val_feature), axis=1)
    train_target1 = train_data.diabetes.iloc[train_idx]
    train_target2 = train_data.diabetes.iloc[val_idx]
    cat_model.fit(train_feat1, train_target1, cat_features=cat_feature_inds)
    print('Train auc', roc_auc_score(train_target1, cat_model.predict_proba(train_feat1)[:, 1]))
    print('Test auc', roc_auc_score(train_target2, cat_model.predict_proba(train_feat2)[:, 1]))
    scores.append(roc_auc_score(train_target2, cat_model.predict_proba(train_feat2)[:, 1]))

print("The average test auc is {0}".format(np.mean(scores)))


cat_model = catboost.CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=1,
        eval_metric='F1',
        random_seed=4 * 100 + 6)

# cat_model.fit(train_data[cols], train_data.diabetes, cat_features=cat_feature_inds)
cat_model.fit(np.concatenate((train_data.as_matrix(cols), conv1[:, 0:20]), axis=1), train_data["diabetes"])
# roc_auc_score(test_data.diabetes, cat_model.predict_proba(test_data[cols])[:, 1])
roc_auc_score(test_data.diabetes, cat_model.predict_proba(
    np.concatenate((test_data.as_matrix(cols), conv1_test[:, 0:20]), axis=1))[:, 1])
