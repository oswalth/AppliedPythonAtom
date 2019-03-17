#!/usr/bin/env python
# coding: utf-8

from homeworks.homework_03.hw3_hashmap import HashMap
# from hw3_hashmap import HashMap


class HashSet(HashMap):
    PRESENT = 'value'

    def __init__(self):
        super().__init__()

    def get(self, key, default_value=None):
        # TODO достаточно переопределить данный метод
        if super().get(key):
            return True
        return False

    def put(self, key, value=None):
        # TODO метод put, нужно переопределить данный метод
        super().put(key, self.PRESENT)

    def intersect(self, another_hashset):
        # TODO метод, возвращающий новый HashSet
        #  элементы - пересечение текущего и другого
        intersection = HashSet()
        for element in self.values():
            if another_hashset.get(element):
                intersection.put(element)
        return intersection

    def values(self):
        values = []
        for bucket in self.buckets:
            if bucket:
                for element in bucket:
                    a = element.key
                    values.append(a)
        return values
