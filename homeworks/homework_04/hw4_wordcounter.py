#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Process, Manager, Queue, Pool
import os


def process_file(q_files, q_objects):
    filename = q_files.get()
    with open('homeworks/homework_04/test_data/' + filename, encoding='utf-8') as f_obj:
        words = f_obj.read().split()
    q_objects.put((filename, len(words)))


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
    q_files = m.Queue()
    q_objects = m.Queue()
    for f in files:
        q_files.put(f)
    container = dict()
    with Pool(3) as p:
        for _ in range(q_files.qsize()):
            p.apply(process_file, (q_files, q_objects))
    for f in files:
        tmp = q_objects.get()
        container[tmp[0]] = tmp[1]
    container['total'] = sum(container.values())
    return container


if __name__ == "__main__":
    result = word_count_inference('homeworks/homework_04/test_data')
    print(result)
