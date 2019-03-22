from multiprocessing import Manager, Pool, Process
import os
import time

def process_file(q, contains):
    filename = q.get()
    with open('homeworks/homework_04/test_data/' + filename, encoding='utf-8') as f_obj:
        words = f_obj.read().split()
    contains[filename] = len(words)

def getpid(n):
    time.sleep(2)
    return os.getpid()

contains = {}
with Pool(3) as p:
    print(p.map(getpid, range(5)))


files = os.listdir(path_to_dir)
with Pool(3) as p:
    p.map(process_file, files)