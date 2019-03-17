#!/usr/bin/env python
# coding: utf-8

from homeworks.homework_03.hw3_hashmap import HashMap


class HashSet(HashMap):

    def __init__(self):
        # TODO Сделать правильно =)
        raise NotImplementedError

    def get(self, key, default_value=None):
        # TODO достаточно переопределить данный метод
        if super().get(key):
            return True
        return False

    def put(self, key, value=None):
        # TODO метод put, нужно переопределить данный метод
        super().put(key, value)

    def intersect(self, another_hashset):
        # TODO метод, возвращающий новый HashSet
        #  элементы - пересечение текущего и другого
        intersection = HashSet()
        for element in self.values():
            if another_hashset.get(element):
                intersection.put(element)
        return intersection
