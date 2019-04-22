#!/usr/bin/env python
# coding: utf-8
import numpy as np
from metrics import logloss, presicion, recall, roc_auc
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import make_classification


class LogisticRegression:
    def __init__(
        self,
        lambda_coef=0.1,
        regulatization=None,
        alpha=0.5,
        n_iter=10000,
        accuracy=1e-4,
    ):
        """
        LogReg for Binary case
        :param lambda_coef: constant coef for gradient descent step
        :param regulatization: regularizarion type ("L1" or "L2") or None
        :param alpha: regularizarion coefficent
        """
        self.lambda_coef = lambda_coef
        self.regularization = regulatization
        self.alpha = alpha
        self.iter_lim = n_iter
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
        self.X_ext = np.hstack([np.ones((X_train.shape[0], 1)), self.X_train])
        self.y_train = y_train[:, np.newaxis]
        self.n, self.k = self.X_ext.shape
        assert X_train.shape[0] == self.y_train.shape[0]
        self.w = np.random.randn(self.k)[:, np.newaxis]

        self.learned = True
        prev = np.zeros_like(self.w)
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
            cur = self.w
            err = np.sum(np.abs(cur - prev))
            if i and err < self.accuracy:
                break
            prev = np.copy(cur)
        self.coef_ = self.w[1:]
        self.intercept_ = self.w[0]

    def _gradient_descent(self, fine):
        y_hat = self.predict_proba(self.X_ext, fit=True)[:, 1]
        self.w -= self.lambda_coef * 1 / self.n * \
            (self.X_ext.T.dot(y_hat[:, np.newaxis] - self.y_train) + fine)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X_test, threshold=0.5, fit=False):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        if not hasattr(self, 'learned'):
            raise NameError

        probability = self.predict_proba(X_test, fit)[:, 1]
        prediction = probability >= threshold
        return prediction.astype(int)

    def predict_proba(self, X_test, fit=False):
        """
        Predict probability using model.
        :param X_test: test data for predict in
        :return: y_test: predicted probabilities
        """
        if fit:
            sigmoid = self._sigmoid(X_test.dot(self.w))
            return np.hstack((1 - sigmoid, sigmoid))
        X_test = (X_test - self.mean_) / self.std_
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        sigmoid = self._sigmoid(X_test.dot(self.w))
        probabilities = np.hstack((1 - sigmoid, sigmoid))
        return probabilities

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        if not hasattr(self, 'learned'):
            raise NameError
        return self.w


np.random.seed(42)
X, y = make_classification(n_samples=1000, n_classes=2,
                           n_informative=14, random_state=43)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

X1, y1 = make_classification(n_samples=1000, n_classes=2,
                             n_informative=14, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.33, random_state=43)

model = LogisticRegression()
model.fit(X_train, y_train)

my = log_loss(y_train, model.predict(X_train))
test_loss = log_loss(y_test[:, np.newaxis], model.predict(X_test))

lmodel = LR(solver='liblinear')
lmodel.fit(X_train, y_train)


my = roc_auc(y_train, lmodel.predict_proba(X_train)[:, 1])
no = roc_auc_score(y_train, lmodel.predict_proba(X_train)[:, 1])

my1 = roc_auc(y_train1, lmodel.predict_proba(X_train1)[:, 1])
no1 = roc_auc_score(y_train1, lmodel.predict_proba(X_train1)[:, 1])


not_my = log_loss(y_train, lmodel.predict(X_train))
not_test = log_loss(y_test, lmodel.predict(X_test))

not_my1 = logloss(y_train, lmodel.predict(X_train))
not_test1 = logloss(y_test, lmodel.predict(X_test))
