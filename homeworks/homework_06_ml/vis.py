import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols
import sympy


class SequantialLearning:
    def __init__(self, F):
        self.cnt = 0
        self.F = F
        self.n = 0
        self.V = 0
        self.v = np.zeros((F.shape[0], 1))
        self.T = np.zeros((F.shape[0], F.shape[0]))
    
    def _get_F(self, X):
        z = symbols('z')
        res = np.array([])
        for el in self.F:
            if isinstance(el, (int,float)):
                res = np.append(res, el)
                continue
            res = np.append(res, float(el.subs(z,X)))
        res = res[:, np.newaxis]
        res = np.reshape(res, (res.shape[1], res.shape[0]))
        return res

    def fit(self, X, y):
        F_num = self._get_F(X)
        ni = 1
        Vi = np.square(y)
        vi = F_num.T.dot(y)
        Ti = F_num.T.dot(F_num)
        self.n += ni
        self.V += Vi
        self.v += vi
        self.T += Ti
        self.cnt += 1
        
    def predict(self, x_test):
        F_num = self._get_F(x_test)
        return F_num.dot(np.linalg.inv(self.T)).dot(self.v)


sigma = 2
x = np.linspace(-10, 10, 1000)
y = 1 + 2*x + 3*x ** 2 - x ** 3 + np.random.normal(scale=sigma)
z = symbols('z')
F = np.array([1, z, z**2,z**3])
regr = SequantialLearning(F)
x_to_plot = np.zeros_like(x)
x_hat_to_plot = []
y_to_plot = np.zeros_like(y)
y_hat_to_plot = np.zeros_like(y)
for i in range(len(x)):
    regr.fit(x[i], y[i])
    x_to_plot[i] = x[i]
    y_to_plot[i] = y[i]
    if i and i % 9 == 0:
        x_hat_to_plot.append(x[i])
        y_hat_to_plot[i] = regr.predict(x[i])
    
y_hat_to_plot = y_hat_to_plot[y_hat_to_plot != 0]
plt.plot(x_to_plot, y_to_plot, '-r')
plt.plot(x_hat_to_plot, y_hat_to_plot, '--b')
plt.ylim((-20,20))
plt.show()
res = regr.predict(5)
print(res)