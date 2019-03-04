#!/usr/bin/env python
# coding: utf-8


def extract(elements):
    answer = []
    for element in elements:
        if type(element) in unhash:
            answer += extract(element)
        else:
            answer.append(element)
    return answer


def invert_dict(source_dict):
    '''
    Функция которая разворачивает словарь, т.е.
    каждому значению ставит в соответствие ключ.
    :param source_dict: dict
    :return: new_dict: dict
    '''
    if not isinstance(source_dict, dict):
        raise NotImplementedError
    new_dict = dict()
    counter = []
    for key, value in source_dict.items():
        if type(value) not in unhash:
            new_dict[value] = key
        else:
            new_keys, new_values = [], []
            answer = extract(value)
            new_keys += answer
            for k in new_keys:
                new_dict[k] = key
    return new_dict
    raise NotImplementedError
