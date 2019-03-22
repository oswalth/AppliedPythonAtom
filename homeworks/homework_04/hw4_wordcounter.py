#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Process, Manager, Queue
import os


def process_file(q, contains):
    filename = q.get()
    with open('homeworks/homework_04/test_data/' + filename, encoding='utf-8') as f_obj:
        words = f_obj.read().split()
    contains[filename] = len(words)


def word_count_inference(path_to_dir):
    '''
    Метод, считающий количество слов в каждом файле из директории
    и суммарное количество слов.
    Слово - все, что угодно через пробел, пустая строка "" словом не считается,
    пробельный символ " " словом не считается. Все остальное считается.
    Решение должно быть многопроцессным. Общение через очереди.
    :param path_to_dir: путь до директории с файлами
    :return: словарь, где ключ - имя файла, значение - число слов +
        специальный ключ "total" для суммы слов во всех файлах
    '''
    m = Manager()
    try:
        files = os.listdir(path_to_dir)
    except FileNotFoundError:
        print("Directory not found")
        return
    q = Queue()
    for f in files:
        q.put(f)
    tasks = []
    contains = m.dict()
    for _ in range(3):
        task = Process(target=process_file(q, contains))
        tasks.append(task)
        task.start()
    for task in tasks:
        task.join()
    contains['total'] = sum(contains.values())

    return contains

#result = process_file('homeworks/homework_04/1.txt')
result = word_count_inference('homeworks/homework_04/test_data')
"""
q = Queue()
ans = []
files = ['file1', 'file2', 'file3']
for f in files:
    q.put(f)
for i in range(len(files)):
    ans.append(q.get())
print(ans)
"""