#!/usr/bin/env python
# coding: utf-8
import numpy as np
from metrics import logloss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression:
    def __init__(self, lambda_coef=1.0, regulatization=None, alpha=0.5, iter_lim=10000, accuracy=1e-9):
        """
        LogReg for Binary case
        :param lambda_coef: constant coef for gradient descent step
        :param regulatization: regularizarion type ("L1" or "L2") or None
        :param alpha: regularizarion coefficent
        """
        self.lambda_coef = lambda_coef
        self.regularization = regulatization
        self.alpha = alpha
        self.iter_lim = iter_lim
        self.accuracy = accuracy

    def fit(self, X_train, y_train):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        self.mean_ = np.mean(X_train)
        self.std_ = np.std(X_train)
        self.X_train = (X_train - self.mean_) / self.std_
        self.X_ext = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])
        self.y_train = y_train[:, np.newaxis]
        self.n, self.k = self.X_ext.shape
        assert self.X_train.shape[0] == self.y_train.shape[0]
        self.w = (np.random.randn(self.k) / np.sqrt(self.k))[:, np.newaxis]

        errs = np.zeros(self.iter_lim)
        for i in range(self.iter_lim):
            if self.regularization == 'L1':
                fine = self.alpha * np.ones(self.k - 1)
                fine = np.insert(fine, 0, 0)[:, np.newaxis]
            elif self.regularization == "L2":
                fine = self.alpha * self.w[1:]
                fine = np.insert(fine, 0, 0)[:, np.newaxis]
            else:
                fine = 0
            
            self._gradient_descent(fine)
            errs[i] = logloss(self.y_train, self.predict_proba(X_train))
            err = np.abs(errs[i] - errs[i - 1])
            if i and err < self.accuracy:
                break
        self.coef_ = self.w[1:]
        self.intercept_ = self.w[0]

    def _gradient_descent(self, fine):
        y_hat = self.predict_proba(self.X_train)
        self.w -= self.lambda_coef * 1 / self.n * self.X_ext.T.dot(y_hat - self.y_train)


    def predict(self, X_test, threshold=0.5):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        probability = self.predict_proba(X_test)
        prediction = np.clip(probability, )
        if probability > threshold:
            return 1
        return 0

    def predict_proba(self, X_test):
        """
        Predict probability using model.
        :param X_test: test data for predict in
        :return: y_test: predicted probabilities
        """
        X_test = (X_test - self.mean_) / self.std_
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        res = 1 / (1 - np.e ** (X_test.dot(self.w)))
        return res

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        return self.w


np.random.seed(42)
df = load_breast_cancer()
x, y = df.data, df.target
x_train, x_test, y_train, y_test = train_test_split(x, y) 
model = LogisticRegression()
model.fit(x_train, y_train)

lmodel = LR()
lmodel.fit(x_train, y_train)

print('ok')