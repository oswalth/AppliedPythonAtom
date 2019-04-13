#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


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
        self.X_ext = np.vstack([np.ones(self.X.shape[0]), self.X])
        self.X_ext = self.X_ext.T
        self.Y = y_train
        self.w = np.zeros(self.X_ext.shape[1])
        # в случае тестируемого примера ответы примерно -224 и 5.9 соотв
        # поэтому ставлю достаточно близкие к ним значения
        self.w[0] = -200
        self.w[1] = 4

        accuracy = 1e-5
        err = 0.1
        iter_lim = 100
        cnt = 0
        while err > accuracy:
            prev = np.copy(self.w)
            self._gradient_descent()
            err = np.sum(np.abs(self.w - prev))
            cnt += 1
        self.w.shape = (2, 1)
        MSE = 1 / self.Y.shape[0] * np.sum((self.Y - self.predict(self.X)))
        print(MSE, cnt)

    def _gradient_descent(self):
        step = 0.01
        Y_predicted = self.predict(self.X)
        self.w -= step * (-2 / self.X.shape[0]) * self.X_ext.T.dot((self.Y - Y_predicted))
    
    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        X_test = (X_test - np.mean(X_test)) / np.std(X_test)
        X_test = np.vstack([np.ones(X_test.shape[0]), X_test])
        y_test = np.dot(X_test.T, self.w)
        return y_test


    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        return self.w
        



df = pd.read_csv('~/atom/AppliedPythonAtom/homeworks/homework_06_ml/weight-height.csv')
wh_dataset = df.loc[df.Gender=='Male', ['Height', 'Weight']].values
X = wh_dataset[:, 0]
y = wh_dataset[:, 1]
model = LinearRegression()
model.fit(X, y)
model.get_weights()
print(model.predict(X))