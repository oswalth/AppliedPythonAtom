import sys
from checking import get_enc, get_format
from tab_process import find_max, from_json
from print_table import output, output_h
# Ваши импорты


def get_data(filename):
    try:
        with open(filename) as f_obj:
            pass
    except FileNotFoundError:
        print("Файл не валиден")
        return None
    if filename:
        enc = get_enc(filename)
        if not enc:
            print("Формат не валиден")
            return None
        format = get_format(filename, enc)
        if not format:
            print("Формат не валиден")
            return None
        with open(filename, encoding=enc) as f_obj:
            data = f_obj.read()
            if format == 'json':
                data = from_json(f_obj)
            elif format == 'tsv':
                data = data.strip().split('\n')
                data = [line.split('\t') for line in data]
                # data = [line.strip().split('\t') for line in f_obj]
        width, col_numb = find_max(data)
        if not width:
            print("Формат не валиден")
            return None
        header = 1
        print('-' * (sum(width) + 5 * col_numb + 1))
        for row in data:
            if header:
                (output_h(row, width))
            else:
                (output(row, width))
            header = 0
        print('-' * (sum(width) + 5 * col_numb + 1), end="")
    else:
        print("Файл не валиден")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Файл не валиден")
        raise SystemExit
    filename = sys.argv[1]
    get_data(filename)
