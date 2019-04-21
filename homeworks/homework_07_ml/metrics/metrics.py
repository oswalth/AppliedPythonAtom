#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sklearn.metrics as metrics


def logloss(y_true, y_pred, eps=1e-10):
    """
    logloss
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated probabilities
    :return: loss
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    ll = y_true.T.dot(np.log(y_pred)) + (1 - y_true).T.dot(np.log(1 - y_pred))
    return -ll / y_pred.shape[0]


def accuracy(y_true, y_pred):
    """
    Accuracy
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated class values
    :return: loss
    """
    diff = y_pred - y_true
    return 1 - (float(np.count_nonzero(diff)) - len(diff))


def presicion(y_true, y_pred):
    """
    presicion
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated class values
    :return: loss
    """
    y_pred = np.clip(y_pred, 0.5, 2)
    diff = y_pred - y_true
    TP = len(diff[diff == 0])
    FP = len(diff[diff == 1])
    return TP / (TP + FP)


def recall(y_true, y_pred):
    """
    presicion
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated class values
    :return: loss
    """
    y_pred = np.clip(y_pred, 0.5, 2)
    diff = y_pred - y_true
    TP = len(diff[diff == 0])
    FN = len(diff[diff == -0.5])
    return TP / (TP + FN)


def FPR(y_true, y_pred):
    """
    False positive ratio
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated class values
    :return: FPR
    """
    y_pred = np.clip(y_pred, 0.5, 2)
    diff = y_pred - y_true
    FP = len(diff[diff == 1])
    TN = len(diff[diff == 0.5])
    return FP / (FP + TN)


def roc_auc(y_true, y_pred):
    """
    roc_auc
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated probabilities
    :return: loss
    """
    TPR_ = []
    FPR_ = []
    treshold = 0
    for _ in range(100):
        tmp = (y_pred >= treshold).astype(int)
        TPR_.append(recall(y_true, tmp))
        FPR_.append(FPR(y_true, tmp))
        treshold += 0.01
        treshold = np.round(treshold, 2)
    roc_auc = np.abs(np.trapz(TPR_, FPR_))
    return roc_auc