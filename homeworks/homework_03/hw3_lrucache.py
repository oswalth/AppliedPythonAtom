#!/usr/bin/env python
# coding: utf-8
from collections import deque
import time


class LRUCacheDecorator:

    def __init__(self, maxsize, ttl):
        '''
        :param maxsize: максимальный размер кеша
        :param ttl: время в млсек, через которое кеш
                    должен исчезнуть
        '''
        # TODO инициализация декоратора
        #  https://www.geeksforgeeks.org/class-as-decorator-in-python/
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = deque()
        self.time = time.time()

    def _put(self, res, *args, **kwargs):
        if len(self.cache) == self.maxsize:
            self.cache.pop()
            self.cache.appendleft({'args': (args, kwargs), 'result': res})
        else:
            self.cache.appendleft({'args': (args, kwargs), 'result': res})

    def _reset(self):
        if self.ttl and time.time() - self.time > self.ttl:
            self.cache = deque()

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            self._reset()
            for element in self.cache:
                if element['args'] == (args, kwargs):
                    self.cache.remove(element)
                    self.cache.appendleft(element)
                    return element['result']
            res = func(*args, **kwargs)
            self._put(res, *args, *kwargs)
            return res
        self.time = time.time()
        return wrapped
