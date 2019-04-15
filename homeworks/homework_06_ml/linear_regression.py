#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from metrics import mse, mae, r2_score


class LinearRegression:
    def __init__(
            self,
            lambda_coef=0.001,
            regulatization=None,
            alpha=0.5,
            accuracy=1e-9,
            iter_lim=100000):
        """
        :param lambda_coef: constant coef for gradient descent step
        :param regulatization: regularizarion type ("L1" or "L2") or None
        :param alpha: regularizarion coefficent
        """
        self.lambda_coef = lambda_coef
        self.regulatization = regulatization
        self.alpha = alpha
        self.accuracy = accuracy
        self.iter_lim = iter_lim

    def fit(self, X_train, y_train):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        self.mean_ = np.mean(X_train)
        self.std_ = np.std(X_train)
        self.X = (X_train - self.mean_) / self.std_
        self.X_ext = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
        self.Y = y_train[:, np.newaxis]
        self.n, self.k = self.X_ext.shape
        self.w = (np.random.randn(self.k) / np.sqrt(self.k))[:, np.newaxis]

        errs = np.zeros(self.iter_lim)
        for i in range(self.iter_lim):
            if self.regulatization == 'L1':
                fine = self.alpha * np.ones(self.k - 1)
                fine = np.insert(fine, 0, 0)[:, np.newaxis]
            elif self.regulatization == "L2":
                fine = self.alpha * self.w[1:]
                fine = np.insert(fine, 0, 0)[:, np.newaxis]
            else:
                fine = 0
            self._gradient_descent(fine)
            errs[i] = mean_squared_error(self.Y, self.predict(X_train))
            err = np.abs(errs[i] - errs[i - 1])
            if i and err < self.accuracy:
                break
        self.coef_ = self.w[1:]
        self.intercept_ = self.w[0]

    def _gradient_descent(self, fine):
        Y_predicted = self.predict(self.X)
        self.w -= self.lambda_coef * (2 / self.n) * \
            (self.X_ext.T.dot((Y_predicted - self.Y)) + fine)

    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        X_test = (X_test - self.mean_) / self.std_
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        y_test = np.dot(X_test, self.w)
        return y_test

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        return self.w


"""
RANDOM_STATE = 42
n_samples = 1000
n_outliers = 50

X, y, coef = make_regression(
    n_samples=n_samples, n_features=1,
    n_informative=1, noise=10,
    coef=True, random_state=RANDOM_STATE
)

# Add outlier data
np.random.seed(RANDOM_STATE)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_STATE
)
my_regression = LinearRegression(regulatization='L2')
my_regression.fit(X_train, y_train)
"""
