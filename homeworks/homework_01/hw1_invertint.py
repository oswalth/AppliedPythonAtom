#!/usr/bin/env python
# coding: utf-8


def reverse(number):
    '''
    Метод, принимающий на вход int и
    возвращающий инвертированный int
    :param number: исходное число
    :return: инвертированное число
    '''

    new = 0
    f = 1
    if number < 0:
        f = -1
    number *= f
    while number > 0:
        last = number % 10
        new = new * 10 + last
        number //= 10

    return new * f
    raise NotImplementedError


print(reverse(-6347))
