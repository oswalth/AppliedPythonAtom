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
    n = y.shape[0]
    return 1 / n * np.sum((y_true - y_hat) ** 2)


def mae(y_true, y_hat):
    """
    Mean absolute error regression loss
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated target values
    :return: loss
    """
    n = y.shape[0]
    return 1 / n * np.sum(np.abs(y_true - y_hat))


def r2_score(y_true, y_hat):
    """
    R^2 regression loss
    :param y_true: vector of truth (correct) target values
    :param y_hat: vector of estimated target values
    :return: loss
    """
    RSS = (np.sum(y_true - y_hat) ** 2)
    TSS = (np.sum(y_true - np.average(y_true))) 
    return 1 -  RSS/TSS 


df = pd.read_csv('./weight-height.csv')
wh_dataset = df.loc[df.Gender=='Male', ['Height', 'Weight']].values
X = wh_dataset[:, 0]
y = wh_dataset[:, 1]
X = np.vstack([np.ones(X.shape), X])
X = X.T
w = (np.linalg.inv((X.T @ X)) @ X.T) @ y
w = w.flatten()
y_hat = (X @ w).reshape(-1)