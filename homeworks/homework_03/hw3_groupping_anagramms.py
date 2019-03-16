#!/usr/bin/env python
# coding: utf-8
from itertools import permutations


def groupping_anagramms(words):
    """
    Функция, которая группирует анаграммы.
    Возвращаем массив, где элементом является массив с анаграмами.
    Пример:  '''Аз есмь строка живу я мерой остр
                За семь морей ростка я вижу рост
                Я в мире сирота
                Я в Риме Ариост'''.split()
                ->
                [
                 ['Аз'], ['есмь', 'семь'],
                 ['строка', 'ростка'], ['живу', 'вижу'],
                 ['я', 'я'], ['мерой', 'морей'],
                 ['остр)'], ['За'], ['рост'], ['Я', 'Я'],
                 ['в', 'в'], ['мире'], ['сирота'],
                 ['Риме'], ['Ариост']
                ]
    :param words: list of words (words in str format)
    :return: list of lists of words
    """
    # TODO: реализовать функцию
    words = words[:]
    output = []
    for word in words:
        f = True
        for o in output:
            if word in o:
                f = False
                continue
        if not f:
            continue
        perms = set([''.join(p) for p in permutations(word.lower())])
        anagramms = []
        for strr in words:
            if strr.lower() in perms:
                anagramms.append(strr)
        output.append(anagramms)

    return output
