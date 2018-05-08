# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC

import _variables
import _data
from _auc import auc

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

train_data, test_data = _data.data()

cols = train_data.columns.values.tolist()
cols.remove("diabetes")

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
scores = []
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(train_data[cols])):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_dat = lgb.Dataset(train_data[cols].iloc[train_idx], train_data["diabetes"].iloc[train_idx])
    valid_dat = lgb.Dataset(train_data[cols].iloc[valid_idx], train_data["diabetes"].iloc[valid_idx])

    gbm = lgb.train(_variables.lgb_params,
                    train_dat,
                    num_boost_round=_variables.num_boost_round,
                    valid_sets=valid_dat,
                    verbose_eval=500,
                    early_stopping_rounds=_variables.early_stopping_rounds,
                    feval=auc)

    tree_feature_train = gbm.predict(train_data[cols].iloc[train_idx, :],
                                     num_iteration=gbm.best_iteration,
                                     pred_leaf=True)
    regr = LogisticRegression(**_variables.LogisticRegParams)
    regr.fit(np.concatenate(
        (tree_feature_train.reshape(-1, 1), train_data.iloc[train_idx, :].as_matrix(cols)), axis=1),
        train_data["diabetes"].iloc[train_idx])

    val_feature = gbm.predict(train_data.iloc[valid_idx, :].as_matrix(cols),
                              num_iteration=gbm.best_iteration,
                              pred_leaf=True
                              )
    probs = regr.predict_proba(
        np.concatenate((val_feature.reshape(-1, 1),
                        train_data.iloc[valid_idx, :].as_matrix(cols)), axis=1))[:, 1]

    scores.append(roc_auc_score(
        train_data.iloc[valid_idx, :].as_matrix(["diabetes"]), probs))

trainD = np.concatenate((train_data.as_matrix(cols), fc1), axis=1)
trainD = lgb.Dataset(train_data[cols], train_data.diabetes)
gbm = lgb.train(_variables.lgb_params,
                trainD,
                num_boost_round=_variables.num_boost_round,
                verbose_eval=500,
                )
gbm.predict(np.concatenate((test_data.as_matrix(cols), fc1_test), axis=1))

train_feat = gbm.predict(train_data.as_matrix(cols),
                        pred_leaf=True)
test_feat = gbm.predict(test_data.as_matrix(cols),
                        pred_leaf=True)
lr = LogisticRegression(**_variables.LogisticRegParams)
lr.fit(np.concatenate((train_feat, train_data.as_matrix(cols)),
                      axis=1), train_data.as_matrix(["diabetes"]))

preds = lr.predict_proba(np.concatenate((test_feat, test_data.as_matrix(cols)),
                      axis=1))[:, 1]
roc_auc_score(test_data.diabetes, preds)


# ============================== STEP 2 =====================================================
rus = RandomUnderSampler(random_state=2018, return_indices=True)
XALL, yALL, idx_resampled = rus.fit_sample(train_data[cols], (train_data["diabetes"] == 1.).astype(int))
yALL = train_data.iloc[idx_resampled]["diabetes"]
XALL = pd.DataFrame(XALL, columns=cols)
test_preds = np.zeros((test_data.shape[0], 5))
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_dat = lgb.Dataset(XALL.iloc[train_idx], yALL.iloc[train_idx])
    valid_dat = lgb.Dataset(XALL.iloc[valid_idx], yALL.iloc[valid_idx])

    gbm = lgb.train(_variables.lgb_params,
                    train_dat,
                    num_boost_round=_variables.num_boost_round,
                    valid_sets=valid_dat,
                    verbose_eval=500,
                    # early_stopping_rounds=_variables.early_stopping_rounds,
                    feval=auc)

    tree_feature_train = gbm.predict(XALL.iloc[train_idx],
                                     # num_iteration=gbm.best_iteration,
                                     pred_leaf=True)
    regr = SVC(**_variables.SVCParams)
    regr.fit(tree_feature_train, yALL.iloc[train_idx])

    test_feature = gbm.predict(test_data[cols],
                               pred_leaf=True,
                               # num_iteration=gbm.best_iteration
                               )
    test_preds[:, cv_idx] = regr.predict_proba(test_feature)[:, 1]
    scores.append(roc_auc_score(test_data.diabetes, test_preds[:, cv_idx]))

step2_preds = test_preds.mean(axis=1)
idx_to_modify = preds < 0.16
preds[idx_to_modify] = step2_preds[idx_to_modify]

