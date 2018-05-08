import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


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
preds = np.zeros(train_data[cols].shape[0])
feature_importance = []
test_preds = np.zeros((test_data[cols].shape[0], 5))
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(train_data[cols])):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_dat = lgb.Dataset(train_data[cols].iloc[train_idx], train_data["diabetes"].iloc[train_idx])
    valid_dat = lgb.Dataset(train_data[cols].iloc[valid_idx], train_data["diabetes"].iloc[valid_idx])

    gbm = lgb.train(_variables.lgb_params,
                    train_dat,
                    num_boost_round=_variables.num_boost_round,
                    valid_sets=valid_dat,
                    verbose_eval=500,
                    # early_stopping_rounds=_variables.early_stopping_rounds,
                    feval=auc)
    test_preds[:, cv_idx] = gbm.predict(test_data[cols])

preds = test_preds.mean(axis=1)


train_D = lgb.Dataset(train_data[cols], train_data["diabetes"])
gbm = lgb.train(_variables.lgb_params,
                train_D,
                num_boost_round=_variables.num_boost_round,
                verbose_eval=500,
                # early_stopping_rounds=_variables.early_stopping_rounds,
                feval=auc)
roc_auc_score(test_data.diabetes, gbm.predict(test_data[cols]))