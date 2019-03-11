import json
from collections import OrderedDict


def find_max(data):
    columns = []
    try:
        col_numb = len(data[0])
        for i in range(col_numb):
            columns.insert(0,  [])
        for row in data:
            for i in range(col_numb):
                columns[i].append(row[i])
    except IndexError:
        return None
    maxs = [len(max(columns[i], key=len)) for i in range(col_numb)]
    return maxs, col_numb

def from_json(f_obj):
    data = []
    try:
        json_list = json.load(f_obj, object_pairs_hook=OrderedDict)
    except ValueError:
        return "Формат не валиден"
    except UnicodeDecodeError:
        return "Формат не валиден"
    for el in json_list:
        entity = []
        for k, v in el.items():
            entity.append(str(v))
        data.append(entity)
    entity = []
    for key in el.keys():
        entity.append(key)
    data.insert(0, entity)
    return data