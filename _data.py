# -*- coding: utf-8 -*-
"""
Created on May 3, 2018
@author: Yue Peng
"""
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


def to_null(x):
    if x in [7., 9., 77., 99.]:
        return np.nan
    else:
        return x


def data():
    dat = pd.read_csv("cdc.csv")

    # NaN preprocessing
    for x in ["ALQ100", "ALQ120U", "ALQ140U",
              "ALQ150", "oldedu", "marital",
              "SMQ020", "SMQ040"]:
        dat[x] = dat[x].apply(to_null)

    # transform procedure
    # cols have more than 30% NA
    features_rm = dat.columns.values[dat.isnull().sum() > 0.3*dat.shape[0]].tolist()

    # remove them
    dat.drop(features_rm, axis=1, inplace=True)

    # remove rows that diabetes is NaN
    dat.dropna(subset=["diabetes"], inplace=True)

    # replace NaN with median
    dat.fillna(dat.median(), inplace=True)

    # remove id
    dat.drop(["sqid"], inplace=True, axis=1)

    cols = dat.columns.values.tolist()
    # remove predictors
    for p in ["pressure", "X"]:
        cols.remove(p)
    dat = dat[cols]
    # train/test split
    train_data = dat[dat.year != 8]
    test_data = dat[dat.year == 8]

    return train_data, test_data

