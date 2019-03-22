#!/usr/bin/env python
# coding: utf-8
from multiprocessing import Process, Manager, Queue, Pool
import os


def process_file(path, filename, container):
    with open(path + '/' + filename, encoding='utf-8') as f_obj:
        words = f_obj.read().split()
    container[filename] = len(words)


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

    container = m.dict()
    tasks = []
    for file in files:
        task = Process(
            target=process_file, args=(
                path_to_dir, file, container))
        tasks.append(task)
        task.start()
    for task in tasks:
        task.join()
    container['total'] = sum(container.values())
    print(container)
    return container
