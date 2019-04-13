from scipy.misc import derivative
import numpy as np

def func(x):
    return x ** 3

print(derivative(func, 2))