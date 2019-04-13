#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse.linalg
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from metrics import mse


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
        self.X_ext = np.vstack([np.ones(self.X.shape[0]), self.X]).T
        self.Y = y_train
        self.n, self.k = self.X_ext.shape
        self.w = np.random.randn(self.k) / np.sqrt(self.k)
        # в случае тестируемого примера ответы примерно -224 и 5.9 соотв
        # поэтому ставлю достаточно близкие к ним значения
        # self.w[0] = -200
        # self.w[1] = 4

        accuracy = 1e-5
        iter_lim = 100
        errs = np.zeros(iter_lim)
        for i in range(iter_lim):
            if self.regulatization == 'L1':
                fine = self.alpha * np.ones(self.k) / 2
            elif self.regulatization == "L2":
                fine = self.alpha * self.w
            else: 
                fine = 0
            self._gradient_descent(fine)
            errs[i] = mse(self.Y, self.predict(X_train))
            if i and np.abs(errs[i] - errs[i - 1]) < accuracy:
                break
        print(errs[i], i)

    def _gradient_descent(self, fine):
        step = 0.01
        Y_predicted = self.predict(self.X)
        self.w -= step * (2 / self.n) * (self.X_ext.T.dot((Y_predicted - self.Y)) + fine)
    
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



df = pd.read_csv('D:/atom/AppliedPythonAtom/homeworks/homework_06_ml/weight-height.csv')
wh_dataset = df.loc[df.Gender=='Male', ['Height', 'Weight']].values
diabetes = datasets.load_diabetes()
X = wh_dataset[:, 0]
y = wh_dataset[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.get_weights())

y_predicted = model.predict(X_test)
MSE1 = mse(y_test, y_predicted)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
y_predicted = model.predict(X_test)
MSE2 = mse(y_test, y_predicted)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
y_predicted = model.predict(X_test)
MSE3 = mse(y_test, y_predicted)
"""
X1 = X_train.reshape(-1, 1)
y1 = y_train.reshape(-1, 1)
my_regr = linear_model.LinearRegression()
my_regr.fit(X1, y1)
y_pred2 = my_regr.predict(X_test.reshape(-1, 1))
MSEe = mse(y_test, y_pred2)
print(my_regr.intercept_, my_regr.coef_)
"""



