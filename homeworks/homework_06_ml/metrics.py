#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


def mse(y_true, y_hat, derivative=False):
    """
    Mean squared error regression loss
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated target values
    :return: loss
    """
    assert y_true.shape == y_hat.shape
    if derivative:
        return 2 * np.sum(y_true - y_hat) / len(y_true)
    return np.average((y_true - y_hat) ** 2)


def mae(y_true, y_hat):
    """
    Mean absolute error regression loss
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated target values
    :return: loss
    """
    assert y_true.shape == y_hat.shape
    return np.average(np.abs(y_true - y_hat))


def r2_score(y_true, y_hat):
    """
    R^2 regression loss
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated target values
    :return: loss
    """
    assert y_true.shape == y_hat.shape
    ESS = np.sum((y_hat - y_true.mean()) ** 2)
    TSS = np.sum((y_true - y_true.mean()) ** 2)
    return (ESS / TSS)
