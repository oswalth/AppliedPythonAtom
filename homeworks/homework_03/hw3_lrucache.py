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

    def put(self, element):
        if len(self.cache) == self.maxsize:
            self.cache.pop()
        self.cache.appendleft(element)

    def reset(self):
        if self.ttl and time.time() - self.time >= self.ttl:
            self.cach = deque()
        return None

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            self.reset()
            for element in self.cache:
                if element['args'] == (args, kwargs):
                    self.cache.remove(element)
                    self.cache.appendleft(element)
                    return element['result']
            res = func(*args, **kwargs)
            self.put({'args': (args, kwargs), 'result': res})
            return res
        self.time = time.time()
        return wrapped
