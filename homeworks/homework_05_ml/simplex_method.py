#!/usr/bin/env python
# coding: utf-8

import numpy as np


def calc_difference(A, c_ext, simplex_diff, basis_coef, b, first=None):
    # считаем симплекс разности для первой итерации
    if first:
        simplex_diff = np.insert(
            simplex_diff, 0, np.sum(
                basis_coef * b, axis=0))
    else:
        simplex_diff[0] = np.sum(basis_coef * b)
    for i in range(1, simplex_diff.shape[0] - 1):
        simplex_diff[i] = np.sum(basis_coef * A[:, i - 1]) - c_ext[i - 1]
    return simplex_diff


def simplex_method(a, b, c):
    """
    Почитать про симплекс метод простым языком:
    * https://  https://ru.wikibooks.org/wiki/Симплекс-метод._Простое_объяснение
    Реализацию алгоритма взять тут:
    * https://youtu.be/gRgsT9BB5-8 (это ссылка на 1-ое из 5 видео).

    Используем numpy и, в целом, векторные операции.

    a * x.T <= b
    c * x.T -> max
    :param a: np.array, shape=(n, m)
    :param b: np.array, shape=(n, 1)
    :param c: np.array, shape=(1, m)
    :return x: np.array, shape=(1, m)
    """
    A = A = np.hstack((a, np.eye(a.shape[0], a.shape[0])))
    basis = [x + 1 for x in range(a.shape[1], A.shape[1])]
    basis_coef = np.array([0] * (A.shape[1] - a.shape[1]))
    c_ext = np.hstack((c, np.zeros(2)))
    simplex_diff = np.zeros(A.shape[1])

    simplex_diff = calc_difference(
        A, c_ext, simplex_diff, basis_coef, b, first=True)
    # продолжаем пока все симплекс разности не будут положительны
    while any(simplex_diff[simplex_diff < 0]):
        # ищем разрешающий столбец
        for col_ind, x in np.ndenumerate(simplex_diff[1:]):
            tmp = simplex_diff[1:]
            max_diff = np.max(np.absolute(tmp[tmp < 0]))
            if x < 0 and abs(x) == max_diff:
                col_ind = col_ind[0]
                break
        # ищем разрешающую строку
        tmp = A[:, col_ind]
        row_min = np.min([b[i] / tmp[i]
                          for i in range(b.shape[0]) if tmp[i] > 0])
        for row_ind, x in np.ndenumerate(tmp):
            if x <= 0:
                continue
            if b[row_ind[0]] / x == row_min:
                row_ind = row_ind[0]
                break
        el = A[row_ind][col_ind] # разрешающий элемент
        recalcA = np.zeros_like(A, dtype=np.float32)
        recalcb = np.zeros_like(b, dtype=np.float32)
        for i in range(len(basis)):
            if basis[i] == basis[row_ind]:
                continue
            for j in range(recalcA.shape[1]):
                recalcA[i][j] = (A[i][j] * el - A[row_ind]
                                 [j] * A[i][col_ind]) / el
            recalcb[i] = (b[i] * el - b[row_ind] * A[i][col_ind]) / el
        recalcb[row_ind] = b[row_ind] / el
        recalcA[row_ind] = A[row_ind] / el
        A = recalcA
        b = recalcb
        basis[row_ind] = col_ind + 1
        basis_coef[row_ind] = c[col_ind]
        simplex_diff = calc_difference(A, c_ext, simplex_diff, basis_coef, b)
    result = np.zeros_like(c_ext)
    for i, el in enumerate(basis):
        if el > a.shape[1]:
            continue
        result[el - 1] = b[i]
    return result[:c.shape[0]]
