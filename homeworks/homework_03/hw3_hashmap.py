#!/usr/bin/env python
# coding: utf-8
from itertools import count


class HashMap:
    '''
    Давайте сделаем все объектненько,
     поэтому внутри хешмапы у нас будет Entry
    '''
    class Entry:

        def __init__(self, key, value):
            '''
            Сущность, которая хранит пары ключ-значение
            :param key: ключ
            :param value: значение
            '''
            self.key = key
            self.value = value

        def get_key(self):
            # TODO возвращаем ключ
            return self.key

        def get_value(self):
            # TODO возвращаем значение
            return self.value

        def __eq__(self, other):
            # TODO реализовать функцию сравнения
            if self.key == other.key:
                return True
            else:
                return False

    def __init__(self, bucket_num=64):
        '''
        Реализуем метод цепочек
        :param bucket_num: число бакетов при инициализации
        '''
        self.size = bucket_num
        self.buckets = [None] * self.size
        self.filled = 0
        self.threshold = 0.75

    def calc_load_ratio(self, filled, total):
        return filled / total

    def get(self, key, default_value=None):
        # TODO метод get, возвращающий значение,
        #  если оно присутствует, иначе default_value
        for bucket in self.buckets:
            if not bucket:
                continue
            for element in bucket:
                if element.key == key:
                    return element.value
        return default_value

    def put(self, key, value):
        # TODO метод put, кладет значение по ключу,
        #  в случае, если ключ уже присутствует он его заменяет
        is_key = key in self
        if not is_key:
            try:
                index = self._get_index(self._get_hash(key))
            except TypeError:
                pass
            else:
                if self.buckets[index] is None:
                    self.buckets[index] = [self.Entry(key, value)]
                    self.filled += 1
                else:
                    self.buckets[index].append(self.Entry(key, value))
                    self.filled += 1
        else:
            for bucket in self.buckets:
                if not bucket:
                    continue
                for element in bucket:
                    if element.key == key:
                        element.value = value
        if self.calc_load_ratio(self.filled, self.size) > self.threshold:
            self._resize()

    def __len__(self):
        # TODO Возвращает количество Entry в массиве
        return self.filled

    def _get_hash(self, key):
        # TODO Вернуть хеш от ключа,
        #  по которому он кладется в бакет
        return(hash(key))

    def _get_index(self, hash_value):
        # TODO По значению хеша вернуть индекс элемента в массиве
        return hash_value % self.size

    def values(self):
        return (
            element.value for bucket in self.buckets if bucket for element in bucket)

    def keys(self):
        return (
            element.key for bucket in self.buckets if bucket for element in bucket)

    def items(self):
        # TODO Должен возвращать итератор пар ключ и значение (tuples)
        return ((element.key, element.value)
                for bucket in self.buckets if bucket for element in bucket)

    def _resize(self):
        self.buckets += [None] * (self.size // 2)
        self.size += self.size // 2

    def __str__(self):
        # TODO Метод выводит "buckets: {}, items: {}"
        raise NotImplementedError

    def __contains__(self, item):
        # TODO Метод проверяющий есть ли объект (через in)
        for bucket in self.buckets:
            if bucket:
                for element in bucket:
                    if element.key == item:
                        return True
        return False
