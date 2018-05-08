# -*- coding: utf-8 -*-
"""
Parameters tuning for CV
"""
import multiprocessing
# ================= LightGBM parameters ====================
lgb_params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.003,
    'num_leaves': 63,
    # 'max_depth': 5,
    'num_threads': multiprocessing.cpu_count() // 2,
    'lambda_l1': 0.01,
    'lambda_l2': 0.01,
    'metric': 'auc',
    'verbose': 0,
    'feature_fraction': .7,
    'feature_fraction_seed': 2,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'bagging_seed': 3,
    'min_data_in_leaf': 10,
    'min_sum_hessian_in_leaf': 11,
    'max_bin': 800,
}

num_boost_round = 3000
early_stopping_rounds = 100


# # ================= Bayesian Ridge Regression ==============
# BayesianRidgeParams = {'n_iter': 1000, 'tol': 1.e-5, 'alpha_1': 1e-06, 'alpha_2': 1e-06,
#                       'lambda_1': 1e-06, 'lambda_2': 1e-06, 'compute_score': False,
#                       'fit_intercept': True, 'normalize': True,
#                       'copy_X': True, 'verbose': False}
#
# # ==================== Ridge Regeression ===================
# RidgeParams = {'alpha': 1.0, 'fit_intercept': True,
#               'normalize': False, 'copy_X': True,
#               'max_iter': None, 'tol': 0.001,
#               'solver': 'auto', 'random_state': None}
#
# # ====================== Elastic Net =======================
# ElasticNetParams = {'alpha': 1.0, 'l1_ratio': 0.1, 'fit_intercept': True,
#             'normalize': False, 'precompute': False, 'max_iter': 1000,
#             'copy_X': True, 'tol': 0.0001, 'warm_start': False,
#             'positive': False, 'random_state': None, 'selection': 'cyclic'}

LogisticRegParams = {"penalty": 'l2', "C": 1}

SVCParams = {'kernel': 'sigmoid', 'degree': 3, 'gamma': 'auto',
            'coef0': 0.0, 'tol': 0.001, 'C': 1.0, 'shrinking': True,
            'cache_size': 200,
            'verbose': True, 'max_iter': -1, 'probability': True}

# HuberParams = {'epsilon': 1.35,'max_iter': 1000, 'alpha': 0.0001,
#               'warm_start': False, 'fit_intercept': True, 'tol': 1e-05}

CatParams = {'iterations': 400, 'learning_rate': 0.03, 'depth': 6,
            'l2_leaf_reg': 1, 'eval_metric': 'AUC', 'random_seed': 2018,
            'thread_count': multiprocessing.cpu_count() // 2,
             "logging_level": "Silent"}

xgbParams = {"max_depth": 6, "gamma":0.2,
             "colsample_bytree": 0.6,
             "min_child_weight": 12,
             "learning_rate": 0.02,
             "objective": "binary:logistic",
             "silent": 1, "eval_metric": "auc"}