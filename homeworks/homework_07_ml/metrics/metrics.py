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
    tp = len(diff[diff == 0])
    fp = len(diff[diff == 1])
    return tp / (tp + fp)


def recall(y_true, y_pred):
    """
    presicion
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated class values
    :return: loss
    """
    y_pred = np.clip(y_pred, 0.5, 2)
    diff = y_pred - y_true
    tp = len(diff[diff == 0])
    fn = len(diff[diff == -0.5])
    return tp / (tp + fn)


def fpr_calc(y_true, y_pred):
    """
    False positive ratio
    :param y_true: vector of truth (correct) class values
    :param y_hat: vector of estimated class values
    :return: FPR
    """
    y_pred = np.clip(y_pred, 0.5, 2)
    diff = y_pred - y_true
    fp = len(diff[diff == 1])
    tn = len(diff[diff == 0.5])
    return fp / (fp + tn)


def roc_auc(y_true, y_pred):
    """
    roc_auc
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated probabilities
    :return: loss
    """
    tpr_list = []
    fpr_list = []
    treshold = 0
    for _ in range(100):
        tmp = (y_pred >= treshold).astype(int)
        tpr_list.append(recall(y_true, tmp))
        fpr_list.append(fpr_calc(y_true, tmp))
        treshold += 0.01
        treshold = np.round(treshold, 2)
    roc_auc = np.abs(np.trapz(tpr_list, fpr_list))
    return roc_auc
