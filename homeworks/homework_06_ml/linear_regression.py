#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse.linalg
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from metrics import mse
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, lambda_coef=1.0, regulatization=None, alpha=0.5):
        """
        :param lambda_coef: constant coef for gradient descent step
        :param regulatization: regularizarion type ("L1" or "L2") or None
        :param alpha: regularizarion coefficent
        """
        self.lambda_coef = lambda_coef
        self.regulatization = regulatization
        self.alpha = alpha

    def fit(self, X_train, y_train):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        self.X = (X_train - np.mean(X_train)) / np.std(X_train)
        self.X_ext = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
        self.Y = y_train
        self.Y.shape = (self.Y.shape[0], 1)
        self.n, self.k = self.X_ext.shape
        self.w = np.random.randn(self.k) / np.sqrt(self.k)
        self.w.shape = (self.w.shape[0], 1)

        accuracy = 1e-7
        iter_lim = 100000
        errs = np.zeros(iter_lim)
        for i in range(iter_lim):
            if self.regulatization == 'L1':
                fine = self.alpha * np.ones((self.k, 1)) / 2
            elif self.regulatization == "L2":
                fine = self.alpha * self.w
            else:
                fine = 0
            self._gradient_descent(fine)
            errs[i] = mse(self.Y, self.predict(X_train))
            err = np.abs(errs[i] - errs[i - 1])
            if i and err < accuracy:
                break
        self.coef_ = self.w[1:]
        self.intercept_ = self.w[0]
        print(err)

    def _gradient_descent(self, fine):
        step = 0.01
        Y_predicted = self.predict(self.X)
        self.w -= step * (2 / self.n) * \
            (self.X_ext.T.dot((Y_predicted - self.Y)) + fine)

    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        X_test = (X_test - np.mean(X_test)) / np.std(X_test)
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        y_test = np.dot(X_test, self.w)
        return y_test

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        return self.w
