import csv
import json


codes = ['utf8', 'utf16', 'cp1251']


def get_enc(filename):
    for enc in codes:
        try:
            with open(filename, encoding=enc) as f_obj:
                f_obj.read()
            return enc
        except UnicodeError:
            pass
    return None


def is_json(filename, enc):
    try:
        with open(filename, encoding=enc) as f_obj:
            json.load(f_obj)
            return True
    except json.JSONDecodeError:
        return False


def is_tsv(filename, enc):
    try:
        with open(filename, encoding=enc) as f_obj:
            csv.reader(f_obj, delimiter="\t")
            return True
    except csv.Error:
        return False


def get_format(filename, enc):
    if is_json(filename, enc):
        return 'json'
    elif is_tsv(filename, enc):
        return "tsv"
